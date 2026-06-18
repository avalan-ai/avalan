from ...agent.orchestrator import Orchestrator
from ...cli.download import DownloadCompleteColumn
from ...cli.theme import (
    Data,
    Spinner,
    Theme,
    TokenRenderDisplayToken,
    TokenRenderDisplayTokenCandidate,
    TokenRenderFrame,
    TokenRenderState,
)
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
from ...event import TOOL_TYPES, Event, EventType
from ...memory.permanent import PermanentMemoryPartition
from ...utils import (
    _j,
    _lf,
    to_json,
    tool_call_diagnostic_payload,
    tool_call_error_payload,
)

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timedelta
from importlib import import_module
from locale import format_string
from logging import Logger
from math import ceil, inf
from re import fullmatch, sub
from textwrap import wrap
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Protocol,
    cast,
)
from uuid import UUID

from humanize import (
    clamp,
    intcomma,
    intword,
    naturalday,
    naturalsize,
    precisedelta,
)
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    SpinnerColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.spinner import Spinner as RichSpinner
from rich.table import Column, Table
from rich.text import Text

if TYPE_CHECKING:
    from numpy import ndarray
else:

    class ndarray:  # noqa: D101
        def __class_getitem__(cls, _: Any) -> Any:
            return Any


def norm(value: object) -> Any:
    numpy_linalg = __import__("numpy.linalg", fromlist=["norm"])
    return cast(Any, numpy_linalg.norm(value))


class _TokenPanelToken(Protocol):
    @property
    def id(self) -> object: ...  # pragma: no cover

    @property
    def token(self) -> str: ...  # pragma: no cover

    @property
    def probability(self) -> float | None: ...  # pragma: no cover


class FancyTheme(Theme):
    _CANONICAL_TOOL_CHANNELS = frozenset(
        {
            "tool_call",
            "tool_execution",
            "tool.call",
            "tool.execution",
        }
    )
    _CANONICAL_TOOL_KIND_PREFIXES = (
        "tool.call.",
        "tool_call.",
        "tool.execution.",
        "tool_execution.",
        "model.continuation.",
        "model_continuation.",
    )

    @property
    def icons(self) -> dict[Data, str]:
        return {
            "access_token_name": ":lock:",
            "agent_id": ":robot:",
            "agent_output": ":robot:",
            "avalan": ":heavy_large_circle:",
            "author": ":briefcase:",
            "bye": ":vulcan_salute:",
            "can_access": ":white_check_mark:",
            "checking_access": ":mag:",
            "created_at": ":calendar:",
            "disabled": ":cross_mark:",
            "download": ":floppy_disk:",
            "download_access_denied": ":exclamation_mark:",
            "download_finished": ":heavy_check_mark:",
            "downloads": ":floppy_disk:",
            "gated": ":key:",
            "inference": ":brain:",
            "input_token_count": ":laptop_computer:",
            "library_name": ":books:",
            "license": ":balance_scale:",
            "likes": ":orange_heart:",
            "memory": ":brain:",
            "model_id": ":name_badge:",
            "model_type": ":robot_face:",
            "no_access": ":no_entry_sign:",
            "parameters": ":abacus:",
            "pipeline_tag": ":gear:",
            "private": ":closed_lock_with_key:",
            "ranking": ":trophy:",
            "path_blobs": ":file_folder:",
            "path_refs": ":file_folder:",
            "path_repository": ":file_folder:",
            "path_snapshot": ":file_folder:",
            "session": ":card_index_dividers:",
            "task_id": ":robot:",
            "total_tokens": ":abacus:",
            "tokens_rate": ":high_voltage:",
            "events": ":bookmark_tabs:",
            "tool_calls": ":hammer:",
            "tool_call_results": ":package:",
            "tool_tokens": ":hammer:",
            "ttft": ":seedling:",
            "ttnt": ":alarm_clock:",
            "ttsr": ":thinking_face:",
            "updated_at": ":calendar:",
            "user": ":hugging_face:",
            "user_input": ":speaking_head:",
            "tags": ":label:",
        }

    @property
    def styles(self) -> dict[Data, str]:
        return {
            "id": "bold",
            "can_access": "green",
            "checking_access": "bright_black blink",
            "created_at": "magenta",
            "downloads": "bright_black",
            "likes": "bright_black",
            "memory": "magenta",
            "memory_embedding_comparison": "dark_orange3",
            "memory_embedding_comparison_similarity": "dark_orange3",
            "memory_embedding_comparison_similarity_high": (
                "bold dark_olive_green3"
            ),
            "memory_embedding_comparison_similarity_middle": "orange_red1",
            "memory_embedding_comparison_similarity_low": "dark_red",
            "model_id": "cyan",
            "no_access": "bold red",
            "parameters": "bold cyan",
            "participant_id": "bold",
            "ranking": "bright_black",
            "session_id": "dark_orange3",
            "score": "dark_orange3",
            "tags": "gray30",
            "updated_at": "magenta",
            "user": "bold cyan",
            "version": "bold",
        }

    @property
    def spinners(self) -> dict[Spinner, str]:
        return {
            "agent_loading": "dots12",
            "cache_accessing": "bouncingBar",
            "connecting": "earth",
            "thinking": "dots",
            "tool_running": "point",
            "downloading": "earth",
        }

    @property
    def quantity_data(self) -> list[str]:
        return ["likes"]

    def flow_run_progress_message(
        self,
        event_type: str,
        *,
        node: str | None = None,
        status: str | None = None,
        attempt: int | None = None,
        flow_name: str | None = None,
    ) -> str:
        _ = status, flow_name
        if node:
            if event_type == "flow_node_started":
                suffix = (
                    f" (attempt {attempt})"
                    if attempt is not None and attempt > 1
                    else ""
                )
                return f"Running [cyan]{node}[/cyan]{suffix}."
            if event_type == "flow_node_retrying":
                suffix = (
                    f" after attempt {attempt}" if attempt is not None else ""
                )
                return f"Retrying [cyan]{node}[/cyan]{suffix}."
            if event_type == "flow_node_completed":
                return f"Finished [cyan]{node}[/cyan]."
            if event_type == "flow_node_failed":
                return f"[cyan]{node}[/cyan] failed."
            if event_type == "flow_node_skipped":
                return f"Skipped [cyan]{node}[/cyan]."
            if event_type == "flow_node_paused":
                return f"Paused [cyan]{node}[/cyan]."
            if event_type == "flow_node_resumed":
                return f"Resumed [cyan]{node}[/cyan]."
            if event_type == "flow_node_cancelled":
                return f"Cancelled [cyan]{node}[/cyan]."
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
        styled_source = _flow_run_styled_mermaid_source(
            mermaid_source,
            node_states=node_states,
            active_nodes=active_nodes,
        )
        diagram = _flow_run_mermaid_renderable(styled_source, console_width)
        status_table = _flow_run_status_table(
            node_states,
            flow_stats=flow_stats,
        )
        progress_body: RenderableType = diagram
        if status_table is not None:
            progress_body = (
                Columns(
                    [diagram, status_table],
                    expand=False,
                    padding=(0, 4),
                )
                if console_width >= 120
                else Group(diagram, Text(""), status_table)
            )
        status_line = Table.grid(padding=(0, 1))
        status_line.add_row(
            RichSpinner(
                "arc",
                text=Text.from_markup(f"[bold]{message}[/bold]"),
                style="cyan",
            )
        )
        return Panel(
            Padding(
                Group(status_line, Text(""), progress_body),
                (1, 2),
            ),
            title="[cyan]Flow progress[/cyan]",
            box=box.SQUARE,
            border_style="cyan",
        )

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
        _i = self._icons
        description_color = (
            "green" if finished else "white" if highlight else "gray62"
        )
        return Panel.fit(
            Padding(
                Group(
                    *_lf(
                        [
                            (
                                f"[{description_color}]"
                                f"{description}[/{description_color}]"
                            ),
                            (
                                _i["author"]
                                + (
                                    f" [bright_black]{author}[/bright_black]"
                                    + " · "
                                    + _i["library_name"]
                                    + f" [bright_black]{library_name}"
                                    + "[/bright_black]"
                                )
                                if highlight
                                else None
                            ),
                        ]
                    )
                )
            ),
            title=_i["task_id"] + f" [cyan]{name}[/cyan]",
            subtitle=_i["model_id"]
            + f" [bright_black]{model_id}[/bright_black]",
            box=box.DOUBLE if highlight else box.SQUARE,
        )

    def agent(
        self,
        agent: Orchestrator,
        *args: object,
        models: list[Model | str],
        cans_access: bool | None = None,
        can_access: bool | None = None,
    ) -> RenderableType:
        _, _f, _i = self._, self._f, self._icons
        permanent_message = agent.memory.permanent_message
        has_session = (
            agent.memory.has_permanent_message
            and permanent_message is not None
            and permanent_message.has_session
        )
        models_group = Group(
            *_lf(
                [
                    _i["model_id"]
                    + " "
                    + ", ".join(
                        [
                            (
                                _("{model_id} ({parameters})").format(
                                    model_id=_f(
                                        "model_id", model.id, icon=False
                                    ),
                                    parameters=_f(
                                        "parameters",
                                        _("{n} params").format(
                                            n=self._parameter_count(
                                                model.parameters
                                            )
                                        ),
                                        icon=False,
                                    ),
                                )
                                if isinstance(model, Model)
                                else str(model)
                            )
                            for model in models
                        ]
                    ),
                    _f(
                        "memory",
                        _j(
                            ", ",
                            _lf(
                                [
                                    (
                                        _("short-term message")
                                        if agent.memory.has_recent_message
                                        else None
                                    ),
                                    (
                                        _(
                                            "long-term message ({driver})"
                                        ).format(
                                            driver=type(
                                                agent.memory.permanent_message
                                            ).__name__
                                        )
                                        if agent.memory.has_permanent_message
                                        else None
                                    ),
                                ]
                            ),
                            empty=_("stateless"),
                        ),
                    ),
                    (
                        _f(
                            "session",
                            " "
                            + _("session: {session_id}").format(
                                session_id=_f(
                                    "session_id",
                                    str(permanent_message.session_id),
                                )
                            ),
                        )
                        if has_session and permanent_message is not None
                        else None
                    ),
                ]
            )
        )
        return Panel(
            models_group,
            title=_f("agent_id", agent.name if agent.name else str(agent.id)),
            box=box.DOUBLE,
        )

    def ask_access_token(self) -> str:
        _ = self._
        return _("Enter your Huggingface access token")

    def ask_delete_paths(self) -> str:
        _ = self._
        return _("Delete selected paths?")

    def ask_login_to_hub(self) -> str:
        _ = self._
        return _("Login to huggingface?")

    def ask_secret_password(self, key: str) -> str:
        _ = self._
        return _("Enter secret for {key}").format(key=key)

    def ask_override_secret(self, key: str) -> str:
        _ = self._
        return _("Secret {key} exists, override?").format(key=key)

    def bye(self) -> RenderableType:
        _, _i = self._, self._icons
        return _i["bye"] + " " + _("bye :)")

    def cache_delete(
        self, cache_deletion: HubCacheDeletion | None, deleted: bool = False
    ) -> RenderableType:
        _, _f, _n, _i = self._, self._f, self._n, self._icons
        if not cache_deletion or (
            not cache_deletion.deletable_blobs
            and not cache_deletion.deletable_refs
            and not cache_deletion.deletable_repos
            and not cache_deletion.deletable_snapshots
        ):
            return Text(_("Nothing found for deletion. No action taken."))

        total_revisions = len(cache_deletion.revisions)
        elements: list[RenderableType] = []
        if not deleted:
            elements.append(
                _n(
                    "{disk_space} of disk space will be freed after deleting "
                    "{total_revisions} revision for {model_id}",
                    "{disk_space} of disk space will be freed after deleting "
                    "{total_revisions} revisions for {model_id}",
                    total_revisions,
                ).format(
                    model_id=_f("model_id", cache_deletion.model_id),
                    total_revisions=total_revisions,
                    disk_space=naturalsize(
                        cache_deletion.deletable_size_on_disk
                    ),
                )
            )

            for field_name, title, deletable_paths in [
                (
                    "path_blobs",
                    _("BLOBs paths"),
                    cache_deletion.deletable_blobs,
                ),
                ("path_refs", _("Refs paths"), cache_deletion.deletable_refs),
                (
                    "path_repository",
                    _("Repository paths"),
                    cache_deletion.deletable_repos,
                ),
                (
                    "path_snapshot",
                    _("Snapshot paths"),
                    cache_deletion.deletable_snapshots,
                ),
            ]:
                if deletable_paths:
                    panel = Panel(
                        Group(
                            *[_f(field_name, path) for path in deletable_paths]
                        ),
                        title=title,
                    )
                    elements.append(Padding(panel, pad=(1, 0, 0, 0)))
        else:
            elements.append(
                _n(
                    "{disk_space} of disk space were freed after deleting "
                    "{total_revisions} revision for {model_id}",
                    "{disk_space} of disk space were freed after deleting "
                    "{total_revisions} revisions for {model_id}",
                    total_revisions,
                ).format(
                    model_id=_f("model_id", cache_deletion.model_id),
                    total_revisions=total_revisions,
                    disk_space=naturalsize(
                        cache_deletion.deletable_size_on_disk
                    ),
                )
            )

        return Group(*elements)

    def cache_list(
        self,
        cache_dir: str,
        cached_models: list[HubCache],
        display_models: list[str] | None = None,
        show_summary: bool = False,
    ) -> RenderableType:
        _ = self._

        if display_models and not show_summary:
            tables: list[RenderableType] = []
            for model_cache in [
                m for m in cached_models if m.model_id in display_models
            ]:
                table = Table(
                    Column(header=_("Revision"), justify="left"),
                    Column(header=_("File name"), justify="left"),
                    Column(
                        header=_("Size on disk"),
                        footer=naturalsize(model_cache.size_on_disk),
                        justify="left",
                    ),
                    Column(header=_("Last accessed"), justify="left"),
                    Column(header=_("Last modified"), justify="left"),
                    title=model_cache.model_id,
                    caption=model_cache.path,
                    show_footer=True,
                    show_header=True,
                    show_edge=True,
                    show_lines=True,
                    border_style="gray58",
                    caption_style="gray58",
                    footer_style="bold cyan",
                )

                last_revision: str | None = None
                for revision, files in model_cache.files.items():
                    for file in files:
                        new_revision = (
                            last_revision is None or revision != last_revision
                        )
                        if new_revision:
                            last_revision = revision

                        summarized_revision = revision[:6]
                        table.add_row(
                            (
                                f"[cyan]{summarized_revision}[/cyan]"
                                if new_revision
                                else (
                                    f"[bright_black]{summarized_revision}"
                                    "[/bright_black]"
                                )
                            ),
                            file.name,
                            f"[bold cyan]{naturalsize(file.size_on_disk)}"
                            "[/bold cyan]",
                            f"[magenta]{naturalday(file.last_accessed)}"
                            "[/magenta]",
                            f"[magenta]{naturalday(file.last_modified)}"
                            "[/magenta]",
                        )

                tables.append(Padding(table, pad=(1, 0, 1, 0)))
            return Group(*tables)
        else:
            filtered_models = (
                [m for m in cached_models if m.model_id in display_models]
                if display_models
                else cached_models
            )
            total_cache_size = sum([m.size_on_disk for m in filtered_models])
            table = Table(
                Column(header=_("Model"), justify="left", no_wrap=True),
                Column(header=_("Revisions"), justify="left"),
                Column(header=_("Total files"), justify="left"),
                Column(
                    header=_("Size on disk"),
                    footer=naturalsize(total_cache_size),
                    justify="left",
                ),
                show_footer=True,
                show_header=True,
                show_edge=True,
                show_lines=True,
                border_style="gray58",
                footer_style="bold cyan",
            )

            for model_cache in filtered_models:
                summarized_revisions = [r[:6] for r in model_cache.revisions]
                table.add_row(
                    model_cache.model_id,
                    f"{model_cache.total_revisions} ([cyan]"
                    + "[/cyan], [cyan]".join(summarized_revisions)
                    + "[/cyan])",
                    intcomma(model_cache.total_files),
                    "[bold cyan]"
                    + naturalsize(model_cache.size_on_disk)
                    + "[/bold cyan]",
                )
            return table

    def download_access_denied(
        self, model_id: str, model_url: str
    ) -> RenderableType:
        _, _i = self._, self._icons
        return Group(
            *_lf(
                [
                    Padding(
                        " ".join(
                            [
                                "[bold red]"
                                + _i["download_access_denied"]
                                + "[/bold red]",
                                "[red]"
                                + _(
                                    "Access denied while trying to download"
                                    " {model_id}"
                                ).format(model_id=model_id)
                                + "[/red]",
                            ]
                        )
                    ),
                    Padding(
                        _(
                            "Ensure you accepted {model_id} license terms:"
                            " {model_url}"
                        ).format(model_id=model_id, model_url=model_url),
                        pad=(1, 0, 0, 0),
                    ),
                ]
            )
        )

    def download_start(self, model_id: str) -> RenderableType:
        _, _i = self._, self._icons
        return Group(
            _i["download"]
            + " "
            + _("Downloading model {model_id}:").format(model_id=model_id),
            "",
        )

    def download_progress(self) -> tuple[str | RenderableType, ...]:
        _ = self._
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
        _, _i = self._, self._icons
        return Padding(
            " ".join(
                [
                    "[bold green]" + _i["download_finished"] + "[/bold green]",
                    _("Downloaded model {model_id} to {path}").format(
                        model_id=model_id, path=path
                    ),
                ]
            )
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
        _ = self._

        event_log = self._events_log(
            events=events,
            events_limit=events_limit,
            include_tokens=include_tokens,
            include_tool_detect=include_tool_detect,
            include_tools=include_tools,
            include_non_tools=include_non_tools,
        )
        panel = (
            Panel(
                _j("\n", event_log),
                title=_("Tool calls") if tool_view else _("Events"),
                title_align="left",
                height=height if height else (2 + (events_limit or 2)),
                padding=(0, 0, 0, 1),
                expand=True,
                box=box.SQUARE,
                border_style="cyan" if tool_view else "gray23",
                style="gray50 on gray15" if tool_view else "gray35 on gray3",
            )
            if event_log
            else None
        )
        return panel

    def logging_in(self, domain: str) -> str:
        _ = self._
        return _("Logging in to {domain}...").format(domain=domain)

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
        input_string_peek: int = 30,
        show_stats: bool = True,
        partition: int | None = None,
        total_partitions: int | None = None,
    ) -> RenderableType:
        _ = self._

        assert (
            total_tokens
            and show_stats
            or (embedding_peek and embeddings.size > 2 * embedding_peek)
        )

        peek_table: Table | None = None
        if embedding_peek and embeddings.size > 2 * embedding_peek:
            input_title = (
                input_string[:input_string_peek] + "…"
                if len(input_string) > input_string_peek
                else input_string
            )
            input_title = sub(r"(\r\n|\r|\n)+", " ", input_title)
            peek_title = (
                _("Part #{partition} of #{partitions}: {text}").format(
                    text=input_title.strip(),
                    partition=partition,
                    partitions=total_partitions,
                )
                if partition and total_partitions
                else input_title
            )
            peek_table = Table(
                title=peek_title,
                show_footer=False,
                show_header=True,
                show_edge=True,
                show_lines=True,
                border_style="gray58",
            )

            if horizontal:
                for i in range(embedding_peek):
                    peek_table.add_column(intcomma(i), justify="center")

                peek_table.add_column(_("..."), justify="center")

                for i in range(
                    embeddings.size - embedding_peek, embeddings.size
                ):
                    peek_table.add_column(intcomma(i), justify="center")

                columns = []
                for i, v in enumerate(embeddings[:embedding_peek]):
                    columns.append(clamp(v, format="{:.4g}"))

                columns.append("")

                for i, v in enumerate(embeddings[-embedding_peek:]):
                    columns.append(clamp(v, format="{:.4g}"))

                peek_table.add_row(*columns)
            else:
                peek_table.add_column(_("position"), justify="right")
                peek_table.add_column(_("value"), justify="left")

                for i, v in enumerate(embeddings[:embedding_peek]):
                    peek_table.add_row(intcomma(i), clamp(v, format="{:.4g}"))

                peek_table.add_row(None, _("..."))

                start_i = embeddings.size - embedding_peek
                for i, v in enumerate(embeddings[-embedding_peek:]):
                    peek_table.add_row(
                        intcomma(start_i + i), clamp(v, format="{:.4g}")
                    )

        stats: Table | None = None
        if show_stats:
            stats = Table(
                show_footer=False,
                show_header=True,
                show_edge=True,
                show_lines=True,
                border_style="gray58",
            )
            stats.add_column(_("Token count"))
            stats.add_column(_("Size"))
            stats.add_column(_("Min"))
            stats.add_column(_("Max"))
            stats.add_column(_("Mean"))
            stats.add_column(_("Std"))
            stats.add_column(_("‖v‖"))
            stats.add_row(
                *(
                    (
                        clamp(x, format="{:.4g}")
                        if isinstance(x, float)
                        else intcomma(x)
                    )
                    for x in (
                        total_tokens,
                        embeddings.size,
                        minv,
                        maxv,
                        meanv,
                        stdv,
                        normv,
                    )
                )
            )

        assert peek_table or stats

        return Group(
            *_lf(
                [
                    Align(peek_table, align="center") if peek_table else None,
                    Align(stats, align="center") if stats else None,
                ]
            )
        )

    def memory_embeddings_comparison(
        self, similarities: dict[str, Similarity], most_similar: str
    ) -> RenderableType:
        assert similarities and most_similar
        _, _f = self._, self._f
        table = Table(
            _("Comparison string"),
            _("Cosine distance"),
            _("L1 distance (Euclidean)"),
            _("L2 distance (Manhattan)"),
            _("Negative dot product"),
            _("Pearson similarity"),
            show_footer=False,
            show_header=True,
            show_edge=True,
            show_lines=True,
            border_style="gray58",
        )
        for compare_string, similarity in similarities.items():
            is_most = compare_string == most_similar
            field_class = (
                "memory_embedding_comparison_similarity_high"
                if similarity.l2_distance <= 0.80
                else (
                    "memory_embedding_comparison_similarity_middle"
                    if similarity.l2_distance <= 1.2
                    else "memory_embedding_comparison_similarity_low"
                )
            )

            table.add_row(
                _f(
                    field_class,
                    compare_string,
                    icon=":trophy: " if is_most else False,
                ),
                _f(
                    field_class,
                    clamp(similarity.cosine_distance, format="{:.4g}"),
                    icon=False,
                ),
                _f(
                    field_class,
                    clamp(similarity.l1_distance, format="{:.4g}"),
                    icon=False,
                ),
                _f(
                    field_class,
                    clamp(similarity.l2_distance, format="{:.4g}"),
                    icon=False,
                ),
                _f(
                    field_class,
                    clamp(similarity.inner_product, format="{:.4g}"),
                    icon=False,
                ),
                _f(
                    field_class,
                    clamp(similarity.pearson, format="{:.4g}"),
                    icon=False,
                ),
            )
        return Align(table, align="center")

    def memory_embeddings_search(
        self,
        matches: list[SearchMatch],
        *args: object,
        match_preview_length: int = 300,
    ) -> RenderableType:
        assert matches
        _, _f = self._, self._f
        table = Table(
            _("Search string"),
            _("Knowledge match"),
            _("L2 distance (Manhattan)"),
            show_footer=False,
            show_header=True,
            show_edge=True,
            show_lines=True,
            border_style="gray58",
        )
        for i, match in enumerate(matches):
            field_class = (
                "memory_embedding_comparison_similarity_high"
                if match.l2_distance <= 0.80
                else (
                    "memory_embedding_comparison_similarity_middle"
                    if match.l2_distance <= 1.2
                    else "memory_embedding_comparison_similarity_low"
                )
            )
            is_most = i == 0
            table.add_row(
                _f(
                    field_class,
                    match.query,
                    icon=":trophy: " if is_most else False,
                ),
                _f(
                    field_class,
                    (
                        match.match
                        if len(match.match) <= match_preview_length
                        else match.match[:match_preview_length] + "..."
                    ),
                    icon=False,
                ),
                _f(
                    field_class,
                    clamp(match.l2_distance, format="{:.4g}"),
                    icon=False,
                ),
            )
        return Align(table, align="center")

    def memory_partitions(
        self,
        partitions: list[TextPartition],
        *args: object,
        display_partitions: int,
    ) -> RenderableType:
        _ = self._
        total_partitions = len(partitions)
        head_count: int = total_partitions
        tail_count: int = 0

        if total_partitions > 2 and total_partitions > display_partitions:
            head_count = ceil((display_partitions - 1) / 2)
            tail_count = display_partitions - head_count

        elements: list[RenderableType] = []

        if head_count:
            for i, partition in enumerate(partitions[:head_count]):
                elements.append(
                    self.memory_embeddings(
                        partition.data,
                        partition.embeddings,
                        total_tokens=partition.total_tokens,
                        minv=partition.embeddings.min().item(),
                        maxv=partition.embeddings.max().item(),
                        meanv=partition.embeddings.mean().item(),
                        stdv=partition.embeddings.std().item(),
                        normv=norm(partition.embeddings).item(),
                        partition=i + 1,
                        total_partitions=total_partitions,
                    )
                )

        if head_count and tail_count:
            elements.append(
                Align(Padding(_("..."), pad=(0, 0, 1, 0)), align="center")
            )

        if tail_count:
            for i, partition in enumerate(partitions[-tail_count:]):
                elements.append(
                    self.memory_embeddings(
                        partition.data,
                        partition.embeddings,
                        total_tokens=partition.total_tokens,
                        minv=partition.embeddings.min().item(),
                        maxv=partition.embeddings.max().item(),
                        meanv=partition.embeddings.mean().item(),
                        stdv=partition.embeddings.std().item(),
                        normv=norm(partition.embeddings).item(),
                        partition=(total_partitions - tail_count) + i + 1,
                        total_partitions=total_partitions,
                    )
                )

        return Group(*elements)

    def model(
        self,
        model: Model,
        *args: object,
        can_access: bool | None = None,
        expand: bool = False,
        summary: bool = False,
    ) -> RenderableType:
        assert (not expand and not summary) or (
            expand ^ summary
        ), "From expand and summary, only one can be set"
        _, _f, _i = self._, self._f, self._icons

        model_lines: list[RenderableType] = _lf(
            [
                _j(
                    " · ",
                    _lf(
                        [
                            _j(
                                " ",
                                _lf(
                                    [
                                        (
                                            _f(
                                                "checking_access",
                                                _("checking access"),
                                            )
                                            if can_access is None
                                            else (
                                                _f(
                                                    "can_access",
                                                    _("access granted"),
                                                )
                                                if can_access
                                                else _f(
                                                    "no_access",
                                                    _("access denied"),
                                                )
                                            )
                                        ),
                                        _f("author", model.author),
                                        (
                                            _f("license", model.license)
                                            if expand and model.license
                                            else None
                                        ),
                                        (
                                            _f("gated", _("gated"))
                                            if model.gated
                                            else None
                                        ),
                                        (
                                            _f("private", _("private"))
                                            if model.private
                                            else None
                                        ),
                                        (
                                            _f("disabled", _("disabled"))
                                            if model.disabled
                                            else None
                                        ),
                                    ]
                                ),
                            ),
                            (
                                (
                                    _i["created_at"]
                                    + " "
                                    + _j(
                                        ", ",
                                        _lf(
                                            [
                                                (
                                                    _f(
                                                        "created_at",
                                                        model.created_at,
                                                        _("created: "),
                                                        icon=False,
                                                    )
                                                    if expand
                                                    else None
                                                ),
                                                _f(
                                                    "updated_at",
                                                    model.updated_at,
                                                    _("updated: "),
                                                    icon=False,
                                                ),
                                            ]
                                        ),
                                    )
                                )
                                if not summary
                                else None
                            ),
                        ]
                    ),
                ),
                (
                    _j(
                        " · ",
                        _lf(
                            [
                                (
                                    _f("model_type", model.model_type)
                                    + (
                                        " ("
                                        + ", ".join(model.architectures)
                                        + ")"
                                        if expand and model.architectures
                                        else ""
                                    )
                                    if model.model_type
                                    else None
                                ),
                                (
                                    _f("library_name", model.library_name)
                                    if model.library_name
                                    else None
                                ),
                                (
                                    _f("inference", model.inference)
                                    if expand and model.inference
                                    else None
                                ),
                                (
                                    _f("pipeline_tag", model.pipeline_tag)
                                    if model.pipeline_tag
                                    else None
                                ),
                            ]
                        ),
                    )
                    if not summary
                    else None
                ),
                (Rule(style="gray30") if expand and model.tags else None),
                (
                    _f("tags", " " + ", ".join(model.tags))
                    if expand and model.tags
                    else None
                ),
            ]
        )
        return Panel(
            Group(*model_lines),
            # Model ID
            title=(
                _f("model_id", model.id)
                + (
                    " "
                    + _j(
                        " ",
                        _lf(
                            [
                                _f(
                                    "parameters",
                                    self._parameter_count(model.parameters),
                                ),
                                (
                                    _f(
                                        "parameter_types",
                                        ", ".join(model.parameter_types),
                                    )
                                    if expand and model.parameter_types
                                    else None
                                ),
                                _("parameters") if expand else _("params"),
                            ]
                        ),
                    )
                )
                if not summary
                else ""
            ),
            # Stats
            subtitle=(
                _j(
                    " ",
                    _lf(
                        [
                            (
                                _f("downloads", model.downloads)
                                if model.downloads
                                else None
                            ),
                            _f("likes", model.likes) if model.likes else None,
                            (
                                _f("ranking", model.ranking)
                                if model.ranking
                                else None
                            ),
                        ]
                    ),
                )
                if expand
                else None
            ),
            box=box.SQUARE,
        )

    def model_display(
        self,
        model_config: ModelConfig | SentenceTransformerModelConfig | None,
        tokenizer_config: TokenizerConfig | None,
        *args: object,
        is_runnable: bool | None = None,
        summary: bool = False,
    ) -> RenderableType:
        _ = self._
        return Group(
            *_lf(
                [
                    (
                        Padding(
                            (
                                self._sentence_transformer_model_config(
                                    model_config,
                                    is_runnable=is_runnable,
                                    summary=summary,
                                )
                                if isinstance(
                                    model_config,
                                    SentenceTransformerModelConfig,
                                )
                                else self._model_config(
                                    model_config,
                                    is_runnable=is_runnable,
                                    summary=summary,
                                )
                            ),
                            pad=(0, 0, 0, 0),
                        )
                        if model_config
                        else None
                    ),
                    Padding(
                        Panel(
                            (
                                self.tokenizer_config(
                                    tokenizer_config, summary=summary
                                )
                                if tokenizer_config
                                else _("No tokenizer settings")
                            ),
                            title=_("Tokenizer settings"),
                            border_style="bright_black",
                        ),
                        pad=(0, 0, 0, 0),
                    ),
                ]
            )
        )

    def _sentence_transformer_model_config(
        self,
        config: SentenceTransformerModelConfig,
        *args: object,
        is_runnable: bool | None,
        summary: bool,
    ) -> RenderableType:
        _ = self._
        config_table = Table(
            Column(header="", justify="right"),
            Column(header="", justify="left"),
            show_footer=False,
            show_header=False,
            show_edge=True,
            show_lines=True,
            border_style="gray58",
        )
        config_table = self._fill_model_config_table(
            config.transformer_model_config,
            config_table,
            is_runnable=is_runnable,
            summary=summary,
        )
        config_table.add_row(_("Backend"), config.backend)
        config_table.add_row(
            _("Similarity function"),
            f"[bold]{config.similarity_function}[/bold]",
        )
        if not summary:
            config_table.add_row(
                _("Truncate dimension"),
                (
                    intcomma(config.truncate_dimension)
                    if config.truncate_dimension
                    else _("No truncation")
                ),
            )

        return Align(config_table, align="center")

    def _model_config(
        self,
        config: ModelConfig,
        *args: object,
        is_runnable: bool | None,
        summary: bool,
    ) -> RenderableType:
        config_table = Table(
            Column(header="", justify="right"),
            Column(header="", justify="left"),
            show_footer=False,
            show_header=False,
            show_edge=True,
            show_lines=True,
            border_style="gray58",
        )
        config_table = self._fill_model_config_table(
            config, config_table, is_runnable=is_runnable, summary=summary
        )
        return Align(config_table, align="center")

    def _fill_model_config_table(
        self,
        config: ModelConfig,
        config_table: Table,
        *args: object,
        is_runnable: bool | None,
        summary: bool,
    ) -> Table:
        _ = self._

        def _int_text(value: int | None) -> str:
            return intcomma(value) if value is not None else _("Unknown")

        config_table.add_row(
            _("Model type"), f"[bold]{config.model_type}[/bold]"
        )

        if is_runnable is not None:
            config_table.add_row(
                _("Runs on this instance"),
                "[bold]" + (_("Yes") if is_runnable else _("No")) + "[/bold]",
            )

        if not summary and config.architectures:
            config_table.add_row(
                _("Architectures"),
                ", ".join([f"[bold]{a}[/bold]" for a in config.architectures]),
            )

        if config.max_position_embeddings:
            config_table.add_row(
                _("Maximum input sequence length"),
                "[bold magenta]"
                + intcomma(config.max_position_embeddings)
                + "[/bold magenta]",
            )
        if not summary:
            config_table.add_row(
                _("Vocabulary size"),
                f"[magenta]{_int_text(config.vocab_size)}[/magenta]",
            )
        config_table.add_row(
            _("Hidden size"),
            f"[magenta]{_int_text(config.hidden_size)}[/magenta]",
        )
        if not summary:
            config_table.add_row(
                _("Number of hidden layers"),
                f"[magenta]{_int_text(config.num_hidden_layers)}[/magenta]",
            )
            config_table.add_row(
                _("Number of attention heads"),
                f"[magenta]{_int_text(config.num_attention_heads)}[/magenta]",
            )
            config_table.add_row(
                _("Number of labels in last layer"),
                f"[magenta]{_int_text(config.num_labels)}[/magenta]",
            )
            if config.loss_type:
                config_table.add_row(
                    _("Type of loss utilized"), config.loss_type
                )

            config_table.add_row(
                _("Returns all attentions"),
                (
                    "[bold]" + _("Yes") + "[/bold]"
                    if config.output_attentions
                    else _("No")
                ),
            )
            config_table.add_row(
                _("Returns all hidden states"),
                (
                    "[bold]" + _("Yes") + "[/bold]"
                    if config.output_hidden_states
                    else _("No")
                ),
            )

            if config.torch_dtype:
                config_table.add_row(
                    _("Weight data type"),
                    f"[green]{config.torch_dtype}[/green]",
                )
            if config.bos_token_id and config.bos_token:
                config_table.add_row(
                    _("Start of stream token"),
                    "[gray50]#"
                    + str(config.bos_token_id)
                    + f"[/gray50] [cyan]{config.bos_token}[/cyan]",
                )
            if config.eos_token_id and config.eos_token:
                config_table.add_row(
                    _("End of stream token"),
                    "[gray50]#"
                    + str(config.eos_token_id)
                    + f"[/gray50] [cyan]{config.eos_token}[/cyan]",
                )
            if config.sep_token_id and config.sep_token:
                config_table.add_row(
                    _("Separation token"),
                    "[gray50]#"
                    + str(config.sep_token_id)
                    + f"[/gray50] [cyan]{config.sep_token}[/cyan]",
                )
            if config.pad_token_id and config.pad_token:
                config_table.add_row(
                    _("Padding token"),
                    "[gray50]#"
                    + str(config.pad_token_id)
                    + f"[/gray50] [cyan]{config.pad_token}[/cyan]",
                )

            if config.prefix:
                config_table.add_row(
                    _("Mandatory beginning prompt"), config.prefix
                )
        return config_table

    def recent_messages(
        self,
        participant_id: UUID,
        agent: Orchestrator,
        messages: list[EngineMessage],
    ) -> RenderableType:
        _, _f, _i = self._, self._f, self._icons
        group = Group(
            *_lf(
                [
                    Panel(
                        str(engine_message.message.content),
                        title=(
                            _i["agent_output"] + " " + _f("id", agent.name)
                            if engine_message.is_from_agent
                            else _i["user_input"]
                            + "  "
                            + _f("participant_id", participant_id)
                        ),
                        title_align="left",
                        expand=True,
                        box=box.SQUARE,
                    )
                    for engine_message in messages
                ]
            )
        )
        return group

    def saved_tokenizer_files(
        self,
        directory_path_or_total: str | int,
        total_files: int | None = None,
    ) -> RenderableType:
        _n = self._n
        if isinstance(directory_path_or_total, int):
            assert total_files is None
            total_files = directory_path_or_total
            directory_path = "."
        else:
            directory_path = directory_path_or_total
            assert total_files is not None
        return Padding(
            _n(
                "Saved {total_files} tokenizer file to {directory_path}",
                "Saved {total_files} tokenizer files to {directory_path}",
                total_files,
            ).format(total_files=total_files, directory_path=directory_path),
            pad=(1, 0, 0, 0),
        )

    def search_message_matches(
        self,
        participant_id: UUID,
        agent: Orchestrator,
        messages: list[EngineMessageScored],
    ) -> RenderableType:
        _, _f, _i = self._, self._f, self._icons
        group = Group(
            *_lf(
                [
                    Panel(
                        str(engine_message.message.content),
                        title=(
                            _i["agent_output"]
                            + " "
                            + _f("id", agent.name or str(agent.id))
                            if engine_message.is_from_agent
                            else _i["user_input"]
                            + "  "
                            + _f("participant_id", participant_id)
                        ),
                        title_align="left",
                        subtitle=_("Matching score: {score}").format(
                            score=_f(
                                "score",
                                clamp(engine_message.score, format="{:.8g}"),
                            )
                        ),
                        subtitle_align="left",
                        expand=True,
                        box=box.SQUARE,
                    )
                    for engine_message in messages
                ]
            )
        )
        return group

    def memory_search_matches(
        self,
        participant_id: UUID,
        namespace: str,
        memories: list[PermanentMemoryPartition],
    ) -> RenderableType:
        _, _f, _i = self._, self._f, self._icons
        group = Group(
            *_lf(
                [
                    Panel(
                        memory.data,
                        title=(
                            _i["memory"]
                            + " "
                            + _f("id", str(memory.memory_id))
                        ),
                        title_align="left",
                        subtitle=_(
                            "Participant: {participant} – Namespace: {ns} –"
                            " Partition: {partition}"
                        ).format(
                            participant=_f(
                                "participant_id", str(participant_id)
                            ),
                            ns=_f("id", namespace),
                            partition=_f("number", memory.partition),
                        ),
                        subtitle_align="left",
                        expand=True,
                        box=box.SQUARE,
                    )
                    for memory in memories
                ]
            )
        )
        return group

    def tokenizer_config(
        self,
        config: TokenizerConfig,
        *args: object,
        summary: bool = False,
    ) -> RenderableType:
        _ = self._

        config_table = Table(
            Column(header="", justify="right"),
            Column(header="", justify="left"),
            show_footer=False,
            show_header=False,
            show_edge=True,
            show_lines=True,
            border_style="gray58",
        )

        config_table.add_row(
            _("Name or path"), f"[bold]{config.name_or_path}[/bold]"
        )
        if not summary:
            if config.tokens:
                config_table.add_row(
                    _("Added tokens"),
                    ", ".join([f"[cyan]{t}[/cyan]" for t in config.tokens]),
                )
            config_table.add_row(
                _("Special tokens"),
                (
                    ", ".join(
                        [
                            f"[cyan]{t}[/cyan]"
                            for t in config.special_tokens or []
                        ]
                    )
                    or _("None")
                ),
            )
        config_table.add_row(
            _("Maximum sequence length"),
            f"[cyan]{config.tokenizer_model_max_length}[/cyan]",
        )
        config_table.add_row(
            _("Fast (rust based)"),
            "[bold]" + _("Yes") + "[/bold]" if config.fast else _("No"),
        )

        return Align(config_table, align="center")

    def tokenizer_tokens(
        self,
        dtokens: list[Token],
        added_tokens: list[str] | None,
        special_tokens: list[str] | None,
        display_details: bool = False,
        current_dtoken: Token | None = None,
        dtokens_selected: list[Token] | None = None,
    ) -> RenderableType:
        return self._token_panels(
            dtokens,
            added_tokens,
            special_tokens,
            display_details=display_details,
            current_dtoken=current_dtoken,
            dtokens_selected=dtokens_selected,
        )

    def _token_panels(
        self,
        dtokens: Sequence[_TokenPanelToken],
        added_tokens: tuple[str, ...] | list[str] | None,
        special_tokens: tuple[str, ...] | list[str] | None,
        *,
        display_details: bool = False,
        current_dtoken: _TokenPanelToken | None = None,
        dtokens_selected: Sequence[_TokenPanelToken] | None = None,
    ) -> RenderableType:
        # Build token panels
        compact_dtokens = True  # For future configurability
        token_panels = [
            Panel(
                Padding(
                    (
                        f"[gray50]#{dtoken.id}[/gray50] {dtoken.token}"
                        if display_details
                        else Text(
                            f"{dtoken.token}",
                            style=(
                                "white on dark_green"
                                if current_dtoken and dtoken == current_dtoken
                                else ""
                            ),
                        )
                    ),
                    pad=(0, 1, 0, 1),
                ),
                padding=(1 if not compact_dtokens else 0, 0),
                border_style=(
                    "green on dark_green"
                    if current_dtoken and dtoken == current_dtoken
                    else (
                        "cyan"
                        if dtokens_selected and dtoken in dtokens_selected
                        else (
                            "magenta"
                            if (
                                (added_tokens and dtoken.token in added_tokens)
                                or (
                                    special_tokens
                                    and dtoken.token in special_tokens
                                )
                            )
                            else "gray30"
                        )
                    )
                ),
                box=box.SQUARE,
            )
            for dtoken in dtokens
        ]

        # Distribute token panels as columns
        columns = Columns(token_panels, equal=False, expand=False, padding=0)
        return Panel(
            Align(columns, align="center"),
            box=box.MINIMAL,
        )

    def display_image_entities(
        self, entities: list[ImageEntity], sort: bool
    ) -> RenderableType:
        _ = self._
        table = Table(
            Column(header=_("Label"), justify="left"),
            Column(header=_("Score"), justify="right"),
            Column(header=_("Box"), justify="left"),
            show_footer=False,
            show_header=True,
            show_edge=True,
            show_lines=True,
            border_style="gray58",
        )

        if sort:
            entities.sort(
                key=lambda e: e.score if e.score is not None else -inf,
                reverse=True,
            )

        for entity in entities:
            score = (
                self._f("score", f"{entity.score:.2f}")
                if entity.score is not None
                else "-"
            )
            box = (
                ", ".join(f"{v:.2f}" for v in entity.box)
                if entity.box
                else "-"
            )
            table.add_row(entity.label, score, box)

        return Align(table, align="center")

    def display_image_entity(self, entity: ImageEntity) -> RenderableType:
        _ = self._
        table = Table(
            Column(header=_("Label"), justify="left"),
            show_footer=False,
            show_header=True,
            show_edge=True,
            show_lines=True,
            border_style="gray58",
        )
        table.add_row(entity.label)
        return Align(table, align="center")

    def display_audio_labels(
        self, audio_labels: dict[str, float]
    ) -> RenderableType:
        _ = self._
        table = Table(
            Column(header=_("Label"), justify="left"),
            Column(header=_("Score"), justify="left"),
            show_footer=False,
            show_header=True,
            show_edge=True,
            show_lines=True,
            border_style="gray58",
        )
        for label, score in audio_labels.items():
            score_text = (
                self._f("score", f"{score:.2f}") if score is not None else "-"
            )
            table.add_row(label, score_text)
        return Align(table, align="center")

    def display_image_labels(self, labels: list[str]) -> RenderableType:
        _ = self._
        table = Table(
            Column(header=_("Label"), justify="left"),
            show_footer=False,
            show_header=True,
            show_edge=True,
            show_lines=True,
            border_style="gray58",
        )
        for label in labels:
            table.add_row(label)
        return Align(table, align="center")

    def display_token_labels(
        self, token_labels: list[dict[str, str]]
    ) -> RenderableType:
        _ = self._
        table = Table(
            Column(header=_("Token"), justify="left"),
            Column(header=_("Label"), justify="left"),
            show_footer=False,
            show_header=True,
            show_edge=True,
            show_lines=True,
            border_style="gray58",
        )
        for pair in token_labels:
            for token, label in pair.items():
                table.add_row(token, label)
        return Align(table, align="center")

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
        return tuple(
            self._iter_token_frames(
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
            )
        )

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
        for frame in self._iter_token_frames(
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

    def _iter_token_frames(
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
    ) -> Iterator[TokenRenderFrame]:
        assert isinstance(state, TokenRenderState)
        _ = logits_count, tool_events_limit, start_thinking
        _, _n, _f, _l = self._, self._n, self._f, logger.debug
        model_id = state.model_id
        added_tokens = state.added_tokens
        special_tokens = state.special_tokens
        display_token_size = state.display_token_size
        display_probabilities = state.display_probabilities
        pick = state.pick
        focus_on_token_when = state.focus_on_token_when
        thinking_text_tokens = (
            list(state.reasoning_text_tokens)
            if state.display_reasoning
            else []
        )
        tool_text_tokens = (
            list(state.tool_text_tokens) if state.display_tools else []
        )
        answer_text_tokens = list(state.answer_text_tokens)
        tokens = list(state.display_tokens) or None
        input_token_count = state.input_token_count
        total_tokens = state.total_tokens
        tool_token_count = state.tool_token_count
        tool_running = state.tool_running
        tool_running_spinner = state.tool_running_spinner
        ttft = state.ttft
        ttnt = state.ttnt
        ttsr = state.ttsr
        elapsed = state.elapsed
        event_stats = state.event_stats

        pick_first = ceil(pick / 2) if pick > 1 else pick
        max_width = console_width - wrap_padding
        think_wrapped = FancyTheme._wrap_lines(
            thinking_text_tokens, max_width, skip_blank_lines=True
        )
        tool_wrapped = FancyTheme._wrap_lines(tool_text_tokens, max_width)
        wrapped = FancyTheme._wrap_lines(answer_text_tokens, max_width)

        think_section = (
            think_wrapped[-(think_height - 2 * think_padding) :]
            if think_wrapped and limit_think_height
            else think_wrapped
        )
        think_wrapped_output = (
            "\n".join(think_section).rstrip() if think_section else None
        )
        tokens_rate = total_tokens / elapsed if elapsed > 0 else 0.0

        tool_section = (
            tool_wrapped[-(tool_height - 2 * tool_padding) :]
            if tool_wrapped and limit_tool_height
            else tool_wrapped
        )
        tool_wrapped_output = (
            "\n".join(tool_section).rstrip() if tool_section else None
        )

        wrapped_section = (
            wrapped[-(height - padding) :]
            if wrapped and limit_answer_height
            else wrapped
        )
        wrapped_output = (
            "\n".join(wrapped_section).rstrip() if wrapped_section else None
        )

        dtokens = (
            tokens[-display_token_size:]
            if display_token_size and tokens
            else None
        )
        dtokens_selected = (
            [
                dtoken
                for dtoken in dtokens
                if focus_on_token_when and focus_on_token_when(dtoken)
            ]
            if dtokens
            else None
        )
        dtokens_total_selected = (
            len(dtokens_selected) if dtokens_selected else 0
        )

        # Build think and, EventStats answer panels
        progress_title = " · ".join(
            _lf(
                [
                    _f(
                        "input_token_count",
                        _n(
                            "{total_tokens} token in",
                            "{total_tokens} tokens in",
                            input_token_count,
                        ).format(total_tokens=input_token_count),
                    ),
                    _f(
                        "total_tokens",
                        _n(
                            "{total_tokens} token out",
                            "{total_tokens} tokens out",
                            total_tokens,
                        ).format(total_tokens=total_tokens),
                    ),
                    (
                        _f(
                            "tool_tokens",
                            _n(
                                "{total_tokens} tool token",
                                "{total_tokens} tool tokens",
                                tool_token_count,
                            ).format(total_tokens=tool_token_count),
                        )
                        if tool_token_count
                        else None
                    ),
                    (
                        _f(
                            "ttft",
                            _("ttft: {ttft} s").format(ttft=f"{ttft:.2f}"),
                        )
                        if ttft
                        else None
                    ),
                    (
                        _f(
                            "ttnt",
                            _("ttnt: {ttnt} s").format(ttnt=f"{ttnt:.1f}"),
                        )
                        if ttnt
                        else None
                    ),
                    (
                        _f(
                            "ttsr",
                            _("rt: {ttsr} s").format(ttsr=f"{ttsr:.1f}"),
                        )
                        if ttsr
                        else None
                    ),
                    _f(
                        "tokens_rate",
                        _("{tokens_rate} t/s").format(
                            tokens_rate=f"{tokens_rate:.2f}"
                        ),
                    ),
                    (
                        _f(
                            "events",
                            _n(
                                "{total} event",
                                "{total} events",
                                event_stats.total_triggers,
                            ).format(total=event_stats.total_triggers),
                        )
                        if event_stats
                        else None
                    ),
                    (
                        _f(
                            "tool_calls",
                            _n(
                                "{total} tool call",
                                "{total} tool calls",
                                event_stats.triggers[EventType.TOOL_EXECUTE],
                            ).format(
                                total=event_stats.triggers[
                                    EventType.TOOL_EXECUTE
                                ]
                            ),
                        )
                        if event_stats
                        and EventType.TOOL_EXECUTE in event_stats.triggers
                        and event_stats.triggers[EventType.TOOL_EXECUTE]
                        else None
                    ),
                    (
                        _f(
                            "tool_call_results",
                            _n(
                                "{total} result",
                                "{total} results",
                                event_stats.triggers[EventType.TOOL_RESULT],
                            ).format(
                                total=event_stats.triggers[
                                    EventType.TOOL_RESULT
                                ]
                            ),
                        )
                        if event_stats
                        and EventType.TOOL_RESULT in event_stats.triggers
                        and event_stats.triggers[EventType.TOOL_RESULT]
                        else None
                    ),
                ]
            )
        )
        think_panel = (
            Panel(
                Align(
                    f"[light_pink1]{think_wrapped_output}[/light_pink1]",
                    vertical="top",
                ),
                title=_("{model_id} reasoning").format(
                    model_id=_f("id", model_id)
                ),
                title_align="left",
                subtitle=progress_title if not wrapped_output else None,
                subtitle_align="right",
                height=(
                    think_height + 2 * think_padding
                    if limit_think_height
                    else None
                ),
                padding=think_padding,
                expand=True,
                box=box.SQUARE,
            )
            if think_wrapped_output
            else None
        )
        tool_panel = (
            Panel(
                Align(f"[cyan]{tool_wrapped_output}[/cyan]", vertical="top"),
                title=_("Tool call requests"),
                title_align="left",
                height=(
                    tool_height + 2 * tool_padding
                    if limit_tool_height
                    else None
                ),
                padding=tool_padding,
                expand=True,
                box=box.SQUARE,
                border_style="bright_cyan",
                style="gray35 on gray3",
            )
            if tool_wrapped_output
            else None
        )
        answer_panel = (
            Panel(
                Align(wrapped_output, vertical="top"),
                title=(
                    _("{model_id} response").format(
                        model_id=_f("id", model_id)
                    )
                    if think_wrapped_output is None
                    else None
                ),
                title_align="left",
                subtitle=(
                    progress_title
                    if wrapped_output or not think_wrapped_output
                    else None
                ),
                subtitle_align="right",
                height=height + 2 * padding if limit_answer_height else None,
                padding=padding,
                expand=True,
                box=box.SQUARE,
            )
            if wrapped_output
            else None
        )
        stats_panel = (
            Panel(
                progress_title,
                title=_("Token stats"),
                title_align="left",
                padding=(0, 1),
                expand=True,
                box=box.SQUARE,
                border_style="bright_black",
                style="gray35 on gray3",
            )
            if not think_panel and not answer_panel
            else None
        )

        tool_running_panel: RenderableType | None = None

        if tool_running_spinner and tool_running:
            tool_running_panel = Padding(
                tool_running_spinner, pad=(1, 0, 1, 0)
            )

        # Quick return of no need for token details
        if display_token_size is None or tokens is None:
            quick_renderables: list[RenderableType] = _lf(
                [
                    stats_panel,
                    think_panel,
                    tool_panel,
                    tool_running_panel,
                    answer_panel,
                ]
            )
            yield (
                None,
                Group(*quick_renderables),
            )
            return

        # Deal with token details
        dtokens_selected_last_index_yielded: int | None = None
        yielded_frames = 0
        yield_next_frame = True

        # As long as we need more frames, we keep yielding
        while yield_next_frame:
            current_selected_index: int | None = None
            tokens_distribution_panel: Panel | None = None

            if display_token_size and tokens:
                assert dtokens is not None
                # Pick current token to highlight
                current_data = None
                current_dtoken: TokenRenderDisplayToken | None = None
                current_dtoken_tokens: (
                    tuple[TokenRenderDisplayTokenCandidate, ...] | None
                ) = None
                if display_probabilities and dtokens_selected:
                    current_selected_index = (
                        0
                        if dtokens_selected_last_index_yielded is None
                        else (
                            dtokens_selected_last_index_yielded + 1
                            if dtokens_selected_last_index_yielded + 1
                            < dtokens_total_selected
                            else None
                        )
                    )
                    _l(
                        f"Selected {current_selected_index} selected token for"
                        " yielding, with"
                        f" {dtokens_selected_last_index_yielded} being the"
                        " previous yielded"
                    )

                    current_dtoken = None
                    if current_selected_index is not None:
                        selected_token = dtokens_selected[
                            current_selected_index
                        ]
                        if selected_token.tokens:
                            current_dtoken = selected_token
                            current_dtoken_tokens = selected_token.tokens
                    current_data = (
                        [
                            t.probability
                            for t in current_dtoken_tokens
                            if t.probability is not None
                        ]
                        if current_dtoken_tokens
                        else None
                    )
                    if current_dtoken:
                        _l(
                            f'Selected "{current_dtoken.token}" as '
                            "interesting token, with "
                            + clamp(
                                current_dtoken.probability or 0.0,
                                format="{:.4g}",
                            )
                            + f"and {current_dtoken_tokens}"
                        )

                tokens_panel = self._token_panels(
                    dtokens,
                    added_tokens,
                    special_tokens,
                    display_details=False,
                    current_dtoken=current_dtoken,
                    dtokens_selected=dtokens_selected,
                )

                # Build bar chart with token alternative probabilities
                chart = None
                if display_probabilities:
                    current_symmetric_indices = (
                        FancyTheme._symmetric_indices(current_data)
                        if current_data
                        else None
                    )
                    current_symmetric_data = None
                    if current_data and current_symmetric_indices:
                        current_symmetric_data = [
                            current_data[i]
                            for i in current_symmetric_indices
                            if i < len(current_data)
                        ]
                    labels = (
                        " ".join(
                            [
                                f"{i + 1}"
                                for i in range(len(current_symmetric_data))
                            ]
                        )
                        if current_symmetric_data
                        else " "
                    )
                    chart_height = 5

                    chart_rows: list[str] = []
                    for level in range(chart_height, 0, -1):
                        chart_row = ""
                        for value in current_symmetric_data or [
                            0 for i in range(pick)
                        ]:
                            if value * chart_height >= level:
                                chart_row += "".join(
                                    [
                                        "[green]",
                                        "█ ",
                                        "[/green]",
                                    ]
                                )
                            else:
                                chart_row += "  "
                        chart_rows.append(chart_row)
                    chart_rows.append(f"[gray30]{labels}[/gray30]")

                    chart = Align(
                        Panel(
                            Group(*[row for row in chart_rows]),
                            border_style="gray30",
                        ),
                        align="center",
                    )

                # Split token alternatives to two tables
                dbatch_first_table = None
                dbatch_second_table = None
                if pick > 0 and current_dtoken and current_dtoken_tokens:
                    dtoken_tokens = current_dtoken_tokens
                    max_dtoken = max(
                        dtoken_tokens,
                        key=lambda dtoken: dtoken.probability or 0.0,
                    )

                    if pick_first is None or len(dtoken_tokens) <= pick_first:
                        dbatch_first = dtoken_tokens
                        dbatch_second = None
                    else:
                        dbatch_first = dtoken_tokens[:pick_first]
                        dbatch_second = dtoken_tokens[pick_first:]

                    if dbatch_first:
                        dbatch_first_table = self._tokens_table(
                            dbatch_first, current_dtoken, max_dtoken
                        )

                    if dbatch_second:
                        dbatch_second_table = self._tokens_table(
                            dbatch_second, current_dtoken, max_dtoken
                        )

                # Build token distribution panel
                distribution_renderables: list[RenderableType] = _lf(
                    [
                        tokens_panel,
                        (
                            Align(
                                Panel.fit(
                                    " [gray50]"
                                    + f"#{current_dtoken.id}"
                                    + f"[/gray50] {current_dtoken.token} ",
                                    border_style="green",
                                    padding=0,
                                ),
                                align="center",
                            )
                            if current_dtoken
                            else None
                        ),
                        chart if current_dtoken and chart else None,
                        (
                            Align(
                                Columns(
                                    _lf(
                                        [
                                            dbatch_first_table,
                                            dbatch_second_table,
                                        ]
                                    )
                                ),
                                align="center",
                            )
                            if dbatch_first_table or dbatch_second_table
                            else None
                        ),
                    ]
                )
                tokens_distribution_panel = Panel(
                    (
                        Group(*distribution_renderables)
                        if chart
                        else tokens_panel
                    ),
                    title=_("token distribution"),
                    title_align="left",
                    border_style="bright_black",
                )

            renderables: list[RenderableType] = _lf(
                [
                    stats_panel,
                    think_panel,
                    tool_panel,
                    tool_running_panel,
                    answer_panel,
                    tokens_distribution_panel if tokens else None,
                ]
            )
            yield (current_dtoken, Group(*renderables))

            yielded_frames = yielded_frames + 1

            yield_next_frame = (
                not maximum_frames or yielded_frames < maximum_frames
            ) and current_selected_index is not None
            if yield_next_frame:
                dtokens_selected_last_index_yielded = current_selected_index
                _l("Will continue to yield next frame")

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
        _, _n = self._, self._n
        if not events or events_limit == 0:
            return None

        event_log: list[str] = []
        for event in events:
            payload = cast(Any, event.payload)
            canonical_payload = self._canonical_event_payload(payload)
            if canonical_payload is not None:
                tool_view = self._canonical_event_tool_view(canonical_payload)
                if tool_view:
                    should_include = include_tools and (
                        event.type != EventType.TOOL_DETECT
                        or include_tool_detect
                    )
                else:
                    should_include = include_non_tools and (
                        event.type != EventType.TOKEN_GENERATED
                        or include_tokens
                    )
                if should_include:
                    event_log.append(
                        self._canonical_event_log(event, canonical_payload)
                    )
                continue

            if (
                include_tools
                and event.type in TOOL_TYPES
                and (
                    event.type != EventType.TOOL_DETECT or include_tool_detect
                )
            ) or (
                include_non_tools
                and event.type not in TOOL_TYPES
                and (event.type != EventType.TOKEN_GENERATED or include_tokens)
            ):
                if (
                    event.type == EventType.TOOL_EXECUTE
                    and isinstance(payload, Mapping)
                    and "call" in payload
                ):
                    call = payload["call"]
                    arguments = call.arguments
                    event_log.append(
                        _(
                            "Executing tool {tool} call #{call_id} with"
                            " {total_arguments} arguments: {arguments}."
                        ).format(
                            tool="[gray78]" + call.name + "[/gray78]",
                            call_id="[gray78]"
                            + str(call.id)[:8]
                            + "[/gray78]",
                            total_arguments=len(arguments or []),
                            arguments="[gray78]"
                            + (
                                s
                                if len(s := str(arguments)) <= 50
                                else s[:47] + "..."
                            )
                            + "[/gray78]",
                        )
                    )
                elif (
                    event.type == EventType.TOOL_MODEL_RUN
                    and isinstance(payload, Mapping)
                    and isinstance(payload.get("messages"), list)
                    and "model_id" in payload
                ):
                    messages = payload["messages"]
                    event_log.append(
                        _n(
                            "Running ReACT model {model_id} with"
                            " {total_messages} message",
                            "Running ReACT model {model_id} with"
                            " {total_messages} messages",
                            len(messages),
                        ).format(
                            model_id=payload["model_id"],
                            total_messages=len(messages),
                        )
                    )
                elif (
                    event.type == EventType.TOOL_MODEL_RESPONSE
                    and isinstance(payload, Mapping)
                    and "model_id" in payload
                ):
                    event_log.append(
                        _("Got ReACT response from model {model_id}").format(
                            model_id=payload["model_id"]
                        )
                    )
                elif event.type == EventType.TOOL_PROCESS and payload:
                    calls = cast(list[Any], payload)
                    event_log.append(
                        _n(
                            "Executing {total_calls} tool: {calls}",
                            "Executing {total_calls} tools: {calls}",
                            len(calls),
                        ).format(
                            total_calls=len(calls),
                            calls="[gray78]"
                            + "[/gray78], [gray78]".join(
                                [call.name for call in calls]
                            )
                            + "[/gray78]",
                        )
                    )
                elif event.type == EventType.TOOL_DIAGNOSTIC and payload:
                    diagnostic = self._tool_diagnostic_from_payload(payload)
                    if diagnostic is None:
                        event_log.append(str(event.payload))
                        continue
                    event_log.append(
                        self._tool_diagnostic_log(diagnostic, payload)
                    )
                elif (
                    event.type == EventType.TOOL_RESULT
                    and isinstance(payload, Mapping)
                    and payload
                    and payload.get("result")
                ):
                    result = payload["result"]
                    if isinstance(result, ToolCallDiagnostic):
                        event_log.append(
                            self._tool_diagnostic_log(result, payload)
                        )
                        continue
                    event_log.append(
                        _(
                            "Executed tool {tool} call #{call_id}"
                            " with {total_arguments} arguments."
                            ' Got result "{result}" in'
                            " {elapsed_with_unit}."
                        ).format(
                            tool="[gray78]" + result.call.name + "[/gray78]",
                            elapsed_with_unit="[gray78]"
                            + precisedelta(
                                timedelta(seconds=event.elapsed or 0.0),
                                minimum_unit="microseconds",
                            )
                            + "[/gray78]",
                            call_id="[gray78]"
                            + str(result.call.id)[:8]
                            + "[/gray78]",
                            total_arguments=len(result.call.arguments or []),
                            result=(
                                "[red]"
                                + to_json(tool_call_error_payload(result))
                                + "[/red]"
                                if isinstance(result, ToolCallError)
                                else (
                                    "[spring_green3]"
                                    + to_json(result.result)
                                    + "[/spring_green3]"
                                )
                            ),
                        )
                    )
                else:
                    event_log.append(
                        (
                            "["
                            + precisedelta(
                                timedelta(seconds=event.elapsed or 0.0)
                            )
                            + f"] <{event.type}>: {event.payload}"
                            if event.payload and event.elapsed
                            else (
                                f"[{datetime.utcfromtimestamp(event.started).isoformat(sep=' ', timespec='seconds')}] <{event.type}>: {event.payload}"  # noqa: E501
                                if event.payload and event.started
                                else (
                                    f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] <{event.type}>: {event.payload}"  # noqa: E501
                                    if event.payload
                                    else (
                                        f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}]"  # noqa: E501
                                        f" <{event.type}>"
                                    )
                                )
                            )
                        )
                    )

        if event_log and events_limit:
            event_log = event_log[-events_limit:]

        return event_log

    @staticmethod
    def _canonical_event_payload(
        payload: Any,
    ) -> Mapping[str, Any] | None:
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

    @classmethod
    def _canonical_event_tool_view(
        cls,
        payload: Mapping[str, Any],
    ) -> bool:
        channel = payload.get("channel")
        if channel in cls._CANONICAL_TOOL_CHANNELS:
            return True
        kind = payload.get("kind")
        return isinstance(kind, str) and kind.startswith(
            cls._CANONICAL_TOOL_KIND_PREFIXES
        )

    def _canonical_event_log(
        self,
        _event: Event,
        payload: Mapping[str, Any],
    ) -> str:
        kind = payload.get("kind")
        channel = payload.get("channel")
        details = [
            self._canonical_event_field_detail("correlation", payload),
            self._canonical_event_field_detail("summary", payload),
            self._canonical_event_field_detail("usage", payload),
            self._canonical_event_scalar_detail("terminal_outcome", payload),
            self._canonical_event_scalar_detail("derived", payload),
        ]
        return self._(
            "Canonical event kind={kind} channel={channel}{details}."
        ).format(
            kind=self._canonical_event_value(kind),
            channel=self._canonical_event_value(channel),
            details="".join(_lf(details)),
        )

    @staticmethod
    def _canonical_event_correlation_detail(
        payload: Mapping[str, Any],
    ) -> str:
        correlation = payload.get("correlation")
        if not isinstance(correlation, Mapping):
            return ""
        for key, label in (
            ("tool_call_id", "call"),
            ("model_continuation_id", "continuation"),
        ):
            value = correlation.get(key)
            if isinstance(value, str) and value:
                return f" for {label} #{value[:8]}"
        return ""

    @classmethod
    def _canonical_event_field_detail(
        cls,
        field_name: str,
        payload: Mapping[str, Any],
    ) -> str | None:
        value = payload.get(field_name)
        if not isinstance(value, Mapping):
            return None
        return (
            f" {field_name}="
            f"{cls._canonical_event_value(cast(Any, dict(value)))}"
        )

    @classmethod
    def _canonical_event_scalar_detail(
        cls,
        field_name: str,
        payload: Mapping[str, Any],
    ) -> str | None:
        if field_name not in payload:
            return None
        value = cls._canonical_event_value(payload[field_name])
        return f" {field_name}={value}"

    @staticmethod
    def _canonical_event_value(value: Any) -> str:
        if isinstance(value, str):
            text = value
        elif isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, int | float) or value is None:
            text = "null" if value is None else str(value)
        else:
            text = to_json(value)
        return "[gray78]" + escape(text) + "[/gray78]"

    def _tool_diagnostic_from_payload(
        self, payload: Any
    ) -> ToolCallDiagnostic | None:
        if not isinstance(payload, dict):
            return None
        diagnostic = payload.get("diagnostic")
        if isinstance(diagnostic, ToolCallDiagnostic):
            return diagnostic
        result = payload.get("result")
        if isinstance(result, ToolCallDiagnostic):
            return result
        diagnostics = payload.get("diagnostics")
        if isinstance(diagnostics, list):
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
        self, diagnostic: ToolCallDiagnostic, payload: Any
    ) -> str:
        call = payload.get("call") if isinstance(payload, dict) else None
        tool_name = (
            getattr(call, "name", None)
            or diagnostic.canonical_name
            or diagnostic.requested_name
            or "tool"
        )
        call_id = (
            getattr(call, "id", None) or diagnostic.call_id or diagnostic.id
        )
        diagnostic_payload = tool_call_diagnostic_payload(diagnostic)
        return self._(
            "Tool diagnostic {code} at {stage} for {tool} call #{call_id}:"
            " {message}."
        ).format(
            code="[yellow]" + diagnostic_payload["code"] + "[/yellow]",
            stage="[gray78]" + diagnostic_payload["stage"] + "[/gray78]",
            tool="[gray78]" + str(tool_name) + "[/gray78]",
            call_id="[gray78]" + str(call_id)[:8] + "[/gray78]",
            message="[yellow]" + diagnostic.message + "[/yellow]",
        )

    def _tokens_table(
        self,
        dbatch: Sequence[_TokenPanelToken],
        current_dtoken: _TokenPanelToken | None,
        max_dtoken: _TokenPanelToken,
    ) -> Table:
        _p = self._percentage

        dtable_color = "gray58"
        table = Table(
            show_footer=False,
            show_header=False,
            show_edge=True,
            show_lines=True,
            border_style=dtable_color,
        )
        table.add_column()
        table.add_column(justify="right")
        table.add_column()
        for dtoken in dbatch:
            is_current_token = (
                current_dtoken and current_dtoken.id == dtoken.id
            )
            is_max_dtoken = max_dtoken.id == dtoken.id
            dtoken_color = (
                "bold green"
                if is_current_token and is_max_dtoken
                else (
                    "green"
                    if is_current_token
                    else "cyan" if is_max_dtoken else dtable_color
                )
            )
            table.add_row(
                f"[gray50]#{dtoken.id}[/gray50]",
                f"[{dtoken_color}]{dtoken.token}[/{dtoken_color}]",
                f"[{dtoken_color}]{_p(dtoken.probability or 0.0)}"
                f"[/{dtoken_color}]",
            )
        return table

    def welcome(
        self,
        url: str,
        name: str,
        version: str,
        license: str,
        user: User | None,
    ) -> RenderableType:
        _, _f, _i = self._, self._f, self._icons
        license_text = _("{license} license").format(license=license)
        return Padding(
            Panel(
                Padding(
                    _j(
                        " - ",
                        _lf(
                            [
                                " ".join(
                                    [
                                        _i["avalan"]
                                        + f" [link={url}]{name}[/link]",
                                        f"[version]{version}[/version]",
                                        "[bright_black]"
                                        + _i["license"]
                                        + f" {license_text}[/bright_black]",
                                    ]
                                ),
                                _f("user", user.name) if user else None,
                                (
                                    _f(
                                        "access_token_name",
                                        user.access_token_name,
                                    )
                                    if user
                                    else None
                                ),
                            ]
                        ),
                    )
                ),
                box=box.SQUARE,
            ),
            pad=(0, 0, 0, 0),  # Might bring lower padding back (0,0,1,0)
        )

    def _parameter_count(self, parameters: int | None) -> str:
        _ = self._
        if not parameters:
            return _("N/A")
        return (
            "{:.1f}B".format(parameters / 1e9)
            if parameters >= 1e9
            else intword(parameters)
        )

    @staticmethod
    def _symmetric_indices(data: list[float]) -> list[int]:
        """Sorts data desc so that highest values in center lower at edge"""
        assert data
        n = len(data)
        result = [-1] * n

        left = n // 2 - 1
        right = n // 2

        for i in range(n):
            if i % 2 == 0:
                result[left] = i
                left -= 1
            else:
                result[right] = i
                right += 1
        return result

    @staticmethod
    def _percentage(value: float) -> str:
        p = value * 100
        return (
            format_string("%d%%", p, grouping=True)
            if p == int(p)
            else format_string("%.1f%%", p, grouping=True)
        )

    @staticmethod
    def _wrap_lines(
        text_tokens: list[str], width: int, skip_blank_lines: bool = False
    ) -> list[str]:
        lines: list[str] = []
        output = "".join(text_tokens)
        for line in output.splitlines():
            wrapped_line = wrap(line, width=width)
            if wrapped_line:
                lines.extend(wrapped_line)
            elif not skip_blank_lines:
                lines.append("")
        return lines


def _flow_run_styled_mermaid_source(
    mermaid_source: str,
    *,
    node_states: Mapping[str, str],
    active_nodes: tuple[str, ...],
) -> str:
    lines = [mermaid_source.rstrip()]
    active = set(active_nodes)
    lines.extend(
        [
            (
                "  classDef avalanRunning fill:#ecfeff,stroke:#0f172a,"
                "color:#0f172a,stroke-width:4px"
            ),
            (
                "  classDef avalanCompleted fill:#166534,stroke:#dcfce7,"
                "color:#dcfce7,stroke-width:2px"
            ),
            (
                "  classDef avalanFailed fill:#991b1b,stroke:#fee2e2,"
                "color:#fee2e2,stroke-width:2px"
            ),
            (
                "  classDef avalanSkipped fill:#525252,stroke:#e5e5e5,"
                "color:#e5e5e5,stroke-width:2px"
            ),
            (
                "  classDef avalanPaused fill:#92400e,stroke:#fef3c7,"
                "color:#fef3c7,stroke-width:2px"
            ),
        ]
    )
    for node, state in node_states.items():
        if not _is_safe_flow_node_identifier(node):
            continue
        class_name = _flow_run_node_class(
            "running" if node in active else state
        )
        if class_name is not None:
            lines.append(f"  class {node} {class_name}")
    return "\n".join(lines) + "\n"


def _flow_run_node_class(state: str) -> str | None:
    match state:
        case "running" | "started" | "retrying":
            return "avalanRunning"
        case "succeeded" | "completed":
            return "avalanCompleted"
        case "failed" | "cancelled":
            return "avalanFailed"
        case "skipped":
            return "avalanSkipped"
        case "paused" | "resumed":
            return "avalanPaused"
        case _:
            return None


def _flow_run_mermaid_renderable(
    mermaid_source: str,
    console_width: int,
) -> RenderableType:
    _ = console_width
    try:
        termaid = import_module("termaid")
    except ImportError:
        return Text(mermaid_source, style="bright_black")
    render_rich = getattr(termaid, "render_rich", None)
    if not callable(render_rich):
        return Text(mermaid_source, style="bright_black")
    try:
        return cast(Callable[..., RenderableType], render_rich)(
            mermaid_source,
            theme="default",
        )
    except Exception as exc:
        return Group(
            Text(
                f"Diagram renderer unavailable ({type(exc).__name__}).",
                style="yellow",
            ),
            Text(mermaid_source, style="bright_black"),
        )


_FLOW_RUN_TOTAL_STATS_KEY = "__total__"


def _flow_run_status_table(
    node_states: Mapping[str, str],
    *,
    flow_stats: Mapping[str, Mapping[str, int | float]] | None = None,
) -> RenderableType | None:
    if not node_states:
        return None
    header = _flow_run_stats_header(flow_stats)
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bright_black", no_wrap=True)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="white", no_wrap=True)
    for _ in range(6):
        table.add_column(no_wrap=True)
    for node, state in node_states.items():
        label, style = _flow_run_state_display(state)
        table.add_row(
            label,
            node,
            Text(state, style=style),
            *_flow_run_node_stats_cells(flow_stats, node),
        )
    return Panel(
        Group(header, table),
        title="[cyan]Nodes[/cyan]",
        box=box.SQUARE,
        border_style="gray35",
        padding=(0, 1),
    )


def _flow_run_stats_header(
    flow_stats: Mapping[str, Mapping[str, int | float]] | None,
) -> Panel:
    stats = _flow_run_stats_values(flow_stats, _FLOW_RUN_TOTAL_STATS_KEY)
    table = Table.grid(expand=True, padding=(0, 1))
    for _ in range(5):
        table.add_column(no_wrap=True, ratio=1)
    table.add_row(
        _flow_run_total_cell("nodes", stats["executed_nodes"]),
        _flow_run_total_cell("ok", stats["succeeded_nodes"]),
        _flow_run_total_cell("in", stats["input_tokens"]),
        _flow_run_total_cell("cached", stats["cached_input_tokens"]),
        _flow_run_total_cell("tool", stats["tools_executed"]),
    )
    table.add_row(
        _flow_run_total_cell(
            "time",
            _flow_run_format_duration(stats["elapsed_ms"]),
        ),
        _flow_run_total_cell("fail", stats["failed_nodes"]),
        _flow_run_total_cell(
            "out",
            stats["output_tokens"],
            value_suffix_markup=":fire:",
            value_suffix_style="dim red",
        ),
        _flow_run_total_cell("rsn", stats["reasoning_tokens"]),
        _flow_run_total_cell(
            "avg",
            _flow_run_format_duration(stats["average_node_ms"]),
        ),
    )
    return Panel(
        table,
        title="[cyan]Stats[/cyan]",
        box=box.SQUARE,
        border_style="cyan",
        padding=(0, 1),
    )


def _flow_run_node_stats_cells(
    flow_stats: Mapping[str, Mapping[str, int | float]] | None,
    node: str,
) -> tuple[Text, ...]:
    stats = _flow_run_stats_values(flow_stats, node)
    return (
        _flow_run_metric_cell(
            ":stopwatch:",
            _flow_run_format_duration(stats["elapsed_ms"]),
            width=6,
        ),
        _flow_run_metric_cell(
            ":incoming_envelope:",
            _flow_run_format_number(stats["input_tokens"]),
        ),
        _flow_run_metric_cell(
            ":floppy_disk:",
            _flow_run_format_percentage(
                stats["cached_input_tokens"],
                stats["input_tokens"],
            ),
        ),
        _flow_run_metric_cell(
            ":speech_balloon:",
            _flow_run_format_number(stats["output_tokens"]),
        ),
        _flow_run_metric_cell(
            ":brain:",
            _flow_run_format_number(stats["reasoning_tokens"]),
        ),
        _flow_run_metric_cell(
            ":hammer_and_wrench:",
            _flow_run_format_number(stats["tools_executed"]),
        ),
    )


def _flow_run_stats_values(
    flow_stats: Mapping[str, Mapping[str, int | float]] | None,
    key: str,
) -> Mapping[str, int | float]:
    source = flow_stats.get(key, {}) if flow_stats is not None else {}
    return {
        "elapsed_ms": _flow_run_stats_number(source, "elapsed_ms"),
        "executed_nodes": _flow_run_stats_number(
            source,
            "executed_nodes",
        ),
        "succeeded_nodes": _flow_run_stats_number(
            source,
            "succeeded_nodes",
        ),
        "failed_nodes": _flow_run_stats_number(source, "failed_nodes"),
        "average_node_ms": _flow_run_stats_number(
            source,
            "average_node_ms",
        ),
        "input_tokens": _flow_run_stats_number(source, "input_tokens"),
        "cached_input_tokens": _flow_run_stats_number(
            source,
            "cached_input_tokens",
        ),
        "output_tokens": _flow_run_stats_number(source, "output_tokens"),
        "reasoning_tokens": _flow_run_stats_number(
            source,
            "reasoning_tokens",
        ),
        "tools_executed": _flow_run_stats_number(
            source,
            "tools_executed",
        ),
    }


def _flow_run_stats_number(
    source: Mapping[str, int | float],
    key: str,
) -> int | float:
    value = source.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(value, 0)


def _flow_run_format_number(value: int | float) -> str:
    return intcomma(int(value))


def _flow_run_format_duration(milliseconds: int | float) -> str:
    value = max(float(milliseconds), 0.0)
    if value < 1000:
        return f"{int(value)}ms"
    seconds = value / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{remaining_seconds:02d}s"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h{remaining_minutes:02d}m"


def _flow_run_format_percentage(value: int | float, total: int | float) -> str:
    if total <= 0 or value <= 0:
        return "0%"
    percentage = min((float(value) / float(total)) * 100, 100)
    return f"{percentage:.0f}%"


def _flow_run_metric_cell(
    emoji_markup: str,
    value: str,
    *,
    width: int = 5,
) -> Text:
    text = Text()
    text.append(Text.from_markup(emoji_markup, style="bright_black"))
    text.append(" ")
    text.append(value.rjust(width), style="white")
    return text


def _flow_run_total_cell(
    label: str,
    value: int | float | str,
    *,
    value_suffix_markup: str | None = None,
    value_suffix_style: str | None = None,
) -> Text:
    rendered = (
        _flow_run_format_number(value)
        if isinstance(value, int | float)
        else value
    )
    text = Text(label, style="bright_black")
    text.append(" ")
    text.append(rendered, style="white")
    if value_suffix_markup is not None:
        text.append(" ")
        suffix = (
            Text.from_markup(value_suffix_markup, style=value_suffix_style)
            if value_suffix_style is not None
            else Text.from_markup(value_suffix_markup)
        )
        text.append(suffix)
    return text


def _flow_run_state_display(state: str) -> tuple[str, str]:
    match state:
        case "running" | "started" | "retrying":
            return ">", "bold cyan"
        case "succeeded" | "completed":
            return "+", "green"
        case "failed" | "cancelled":
            return "!", "bold red"
        case "skipped":
            return "-", "bright_black"
        case "paused" | "resumed":
            return "=", "yellow"
        case _:
            return ".", "bright_black"


def _is_safe_flow_node_identifier(node: str) -> bool:
    return fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", node) is not None
