from ...agent.orchestrator import Orchestrator
from ...cli.theme import Data, Spinner, Theme
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
from ...event import Event, EventStats, EventType, TOOL_TYPES
from ...memory.partitioner.text import TextPartition
from ...memory.permanent import Memory
from ...utils import _j, _lf
from datetime import datetime
from humanize import (
    clamp,
    intcomma,
    intword,
    naturalday,
    naturalsize,
    precisedelta,
)
from locale import format_string
from logging import Logger
from math import ceil
from numpy import ndarray
from numpy.linalg import norm
from os import linesep
from re import sub
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    SpinnerColumn,
)
from rich.rule import Rule
from rich.table import Column, Table
from rich.text import Text
from textwrap import wrap
from typing import Callable, Generator
from uuid import UUID


class FancyTheme(Theme):
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
            "ttft": ":seedling:",
            "ttnt": ":alarm_clock:",
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
        *args,
        models: list[Model | str],
        can_access: bool | None,
    ) -> RenderableType:
        _, _f, _i = self._, self._f, self._icons
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
                            [
                                (
                                    _("short-term message")
                                    if agent.memory.has_recent_message
                                    else None
                                ),
                                (
                                    _("long-term message ({driver})").format(
                                        driver=type(
                                            agent.memory.permanent_message
                                        ).__name__
                                    )
                                    if agent.memory.has_permanent_message
                                    else None
                                ),
                            ],
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
                                    str(
                                        agent.memory.permanent_message.session_id
                                    ),
                                )
                            ),
                        )
                        if agent.memory.has_permanent_message
                        and agent.memory.permanent_message.has_session
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
        self, cache_deletion: HubCacheDeletion | None, deleted=False
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
            display_models = (
                [m for m in cached_models if m.model_id in display_models]
                if display_models
                else cached_models
            )
            total_cache_size = sum([m.size_on_disk for m in display_models])
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

            for model_cache in display_models:
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

    def download_progress(self) -> tuple[str | RenderableType]:
        _ = self._
        return (
            SpinnerColumn(),
            (
                "[progress.description]{task.description}"
                "[progress.percentage]{task.percentage:>4.0f}%"
            ),
            BarColumn(bar_width=None),
            "[",
            MofNCompleteColumn(),
            "-",
            TimeElapsedColumn(),
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
    ) -> RenderableType:
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
                    icon=":trophy: " if is_most else None,
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
        *args,
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
                    icon=":trophy: " if is_most else None,
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
        self, partitions: list[TextPartition], *args, display_partitions: int
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
        *args,
        can_access: bool | None = None,
        expand: bool = False,
        summary: bool = False,
    ) -> RenderableType:
        assert (not expand and not summary) or (
            expand ^ summary
        ), "From expand and summary, only one can be set"
        _, _f, _i = self._, self._f, self._icons

        return Panel(
            Group(
                *_lf(
                    [
                        _j(
                            " · ",
                            [
                                _j(
                                    " ",
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
                                    ],
                                ),
                                (
                                    (
                                        _i["created_at"]
                                        + " "
                                        + _j(
                                            ", ",
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
                                            ],
                                        )
                                    )
                                    if not summary
                                    else None
                                ),
                            ],
                        ),
                        (
                            _j(
                                " · ",
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
                                ],
                            )
                            if not summary
                            else None
                        ),
                        (
                            Rule(style="gray30")
                            if expand and model.tags
                            else None
                        ),
                        (
                            _f("tags", " " + ", ".join(model.tags))
                            if expand and model.tags
                            else None
                        ),
                    ]
                )
            ),
            # Model ID
            title=(
                _f("model_id", model.id)
                + (
                    " "
                    + _j(
                        " ",
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
                        ],
                    )
                )
                if not summary
                else ""
            ),
            # Stats
            subtitle=(
                _j(
                    " ",
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
                    ],
                )
                if expand
                else None
            ),
            box=box.SQUARE,
        )

    def model_display(
        self,
        model_config: ModelConfig | SentenceTransformerModelConfig | None,
        tokenizer_config: TokenizerConfig,
        *args,
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
                            self.tokenizer_config(
                                tokenizer_config, summary=summary
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
        *args,
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
        *args,
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
        *args,
        is_runnable: bool | None,
        summary: bool,
    ) -> Table:
        _ = self._
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
                f"[magenta]{intcomma(config.vocab_size)}[/magenta]",
            )
        config_table.add_row(
            _("Hidden size"),
            f"[magenta]{intcomma(config.hidden_size)}[/magenta]",
        )
        if not summary:
            config_table.add_row(
                _("Number of hidden layers"),
                f"[magenta]{intcomma(config.num_hidden_layers)}[/magenta]",
            )
            config_table.add_row(
                _("Number of attention heads"),
                f"[magenta]{intcomma(config.num_attention_heads)}[/magenta]",
            )
            config_table.add_row(
                _("Number of labels in last layer"),
                f"[magenta]{intcomma(config.num_labels)}[/magenta]",
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
    ):
        _, _f, _i = self._, self._f, self._icons
        group = Group(
            *_lf(
                [
                    Panel(
                        engine_message.message.content,
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
        self, directory_path: str, total_files: int
    ) -> RenderableType:
        _n = self._n
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
    ):
        _, _f, _i = self._, self._f, self._icons
        group = Group(
            *_lf(
                [
                    Panel(
                        engine_message.message.content,
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
        self, participant_id: UUID, namespace: str, memories: list[Memory]
    ) -> RenderableType:
        _, _f, _i = self._, self._f, self._icons
        group = Group(
            *_lf(
                [
                    Panel(
                        memory.data,
                        title=(
                            _i["memory"] + " " + _f("id", memory.identifier)
                        ),
                        title_align="left",
                        subtitle=_(
                            "Participant: {participant} – Namespace: {ns}"
                        ).format(
                            participant=_f("participant_id", participant_id),
                            ns=_f("id", namespace),
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
        self, config: TokenizerConfig, *args, summary: bool = False
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
                ", ".join(
                    [f"[cyan]{t}[/cyan]" for t in config.special_tokens]
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
        dtokens_selected: list[Token] = [],
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
                                else None
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
                        if dtoken in dtokens_selected
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
        tool_running_spinner: Spinner | None,
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
        _, _n, _f, _l = self._, self._n, self._f, logger.debug

        # Prepare data for rendering
        output = "".join(text_tokens)
        pick_first = ceil(pick / 2) if pick > 1 else pick
        max_width = console_width - wrap_padding
        think_wrapped, wrapped = [], []
        is_thinking = start_thinking

        for line in output.splitlines():
            if line == "<think>":
                is_thinking = True
                continue
            elif is_thinking and line == "</think>":
                is_thinking = False
                continue

            wrapped_line = wrap(line, width=max_width)
            if is_thinking:
                think_wrapped.extend(wrapped_line)
                think_wrapped.append(linesep)
            else:
                wrapped.extend(wrapped_line)
                wrapped.append(linesep)

        think_section = (
            think_wrapped[-(think_height - 2 * think_padding) :]
            if think_wrapped and limit_think_height
            else think_wrapped
        )
        think_wrapped_output = (
            " ".join(think_section).strip() if think_section else None
        )

        wrapped_section = (
            wrapped[-(height - padding) :]
            if wrapped and limit_answer_height
            else wrapped
        )
        wrapped_output = (
            " ".join(wrapped_section).strip() if wrapped_section else None
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
                        _("{total_tokens} input tokens").format(
                            total_tokens=input_token_count
                        ),
                    ),
                    _f(
                        "total_tokens",
                        _("{total_tokens} output tokens").format(
                            total_tokens=total_tokens
                        ),
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
                            _("ttnt: {ttnt} s").format(ttnt=f"{ttnt:.2f}"),
                        )
                        if ttnt
                        else None
                    ),
                    _f(
                        "tokens_rate",
                        _("{tokens_rate} t/s").format(
                            tokens_rate=f"{total_tokens / ellapsed:.2f}"
                        ),
                    ),
                    (
                        _f(
                            "events",
                            _("{total} events").format(
                                total=event_stats.total_triggers
                            ),
                        )
                        if event_stats
                        else None
                    ),
                    (
                        _f(
                            "tool_calls",
                            _("{total} tool calls").format(
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
                            _("{total} results").format(
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
        think_pannel = (
            Panel(
                f"[bright_black]{think_wrapped_output}[/bright_black]",
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
        answer_panel = (
            Panel(
                wrapped_output,
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

        tool_running_panel: RenderableType | None = None

        if tool_running_spinner and len(tool_event_calls) != len(
            tool_event_results
        ):
            tool_running_panel = Padding(
                tool_running_spinner, pad=(1, 0, 1, 0)
            )

        # Quick return of no need for token details
        if display_token_size is None or tokens is None:
            yield (
                None,
                Group(
                    *_lf(
                        [
                            think_pannel or None,
                            tool_running_panel or None,
                            answer_panel or None,
                        ]
                    )
                ),
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
                # Pick current token to highlight
                current_data = None
                current_dtoken: Token | None = None
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

                    current_dtoken = (
                        dtokens_selected[current_selected_index]
                        if current_selected_index is not None
                        else None
                    )
                    current_data = (
                        [t.probability for t in current_dtoken.tokens]
                        if current_dtoken and current_dtoken.tokens
                        else None
                    )
                    if current_dtoken:
                        _l(
                            f'Selected "{current_dtoken.token}" as '
                            "interesting token, with "
                            + clamp(
                                current_dtoken.probability, format="{:.4g}"
                            )
                            + f"and {current_dtoken.tokens}"
                        )

                tokens_panel = self.tokenizer_tokens(
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
                    current_symmetric_data = (
                        [current_data[i] for i in current_symmetric_indices]
                        if current_data
                        else None
                    )
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
                if pick > 0 and current_dtoken and current_dtoken.tokens:
                    dtoken_tokens = current_dtoken.tokens
                    max_dtoken = max(
                        dtoken_tokens, key=lambda dtoken: dtoken.probability
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
                tokens_distribution_panel = Panel(
                    (
                        Group(
                            *_lf(
                                [
                                    tokens_panel,
                                    (
                                        Align(
                                            Panel.fit(
                                                " [gray50]"
                                                + f"#{current_dtoken.id}"
                                                + "[/gray50]"
                                                f" {current_dtoken.token} ",
                                                border_style="green",
                                                padding=0,
                                            ),
                                            align="center",
                                        )
                                        if current_dtoken
                                        else None
                                    ),
                                    (
                                        chart
                                        if current_dtoken and chart
                                        else None
                                    ),
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
                                        if dbatch_first_table
                                        or dbatch_second_table
                                        else None
                                    ),
                                ]
                            )
                        )
                        if chart
                        else tokens_panel
                    ),
                    title=_("token distribution"),
                    title_align="left",
                    border_style="bright_black",
                )

            yield (
                current_dtoken,
                Group(
                    *_lf(
                        [
                            think_pannel or None,
                            tool_running_panel or None,
                            answer_panel or None,
                            (
                                tokens_distribution_panel
                                if tokens and tokens_distribution_panel
                                else None
                            ),
                        ]
                    )
                ),
            )

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

        event_log: list[str] | None = _lf(
            [
                (
                    _(
                        "Executing tool {tool} call #{call_id} with"
                        " {total_arguments} arguments: {arguments}."
                    ).format(
                        tool="[gray78]"
                        + event.payload["call"].name
                        + "[/gray78]",
                        call_id="[gray78]"
                        + str(event.payload["call"].id)[:8]
                        + "[/gray78]",
                        total_arguments=len(
                            event.payload["call"].arguments or []
                        ),
                        arguments="[gray78]"
                        + (
                            s
                            if len(s := str(event.payload["call"].arguments))
                            <= 50
                            else s[:47] + "..."
                        )
                        + "[/gray78]",
                    )
                    if event.type == EventType.TOOL_EXECUTE
                    else (
                        _n(
                            "Running ReACT model {model_id} with"
                            " {total_messages} message",
                            "Running ReACT model {model_id} with"
                            " {total_messages} messages",
                            len(event.payload["messages"]),
                        ).format(
                            model_id=event.payload["model_id"],
                            total_messages=len(event.payload["messages"]),
                        )
                        if event.type == EventType.TOOL_MODEL_RUN
                        else (
                            _(
                                "Got ReACT response from model {model_id}"
                            ).format(model_id=event.payload["model_id"])
                            if event.type == EventType.TOOL_MODEL_RESPONSE
                            else (
                                _n(
                                    "Executing {total_calls} tool: {calls}",
                                    "Executing {total_calls} tools: {calls}",
                                    len(event.payload),
                                ).format(
                                    total_calls=len(event.payload),
                                    calls="[gray78]"
                                    + "[/gray78], [gray78]".join(
                                        [call.name for call in event.payload]
                                    )
                                    + "[/gray78]",
                                )
                                if event.type == EventType.TOOL_PROCESS
                                else (
                                    _(
                                        "Executed tool {tool} call #{call_id}"
                                        " with {total_arguments} arguments."
                                        ' Got result "{result}" in'
                                        " {ellapsed_with_unit}."
                                    ).format(
                                        tool="[gray78]"
                                        + event.payload["result"].call.name
                                        + "[/gray78]",
                                        ellapsed_with_unit="[gray78]"
                                        + precisedelta(
                                            event.ellapsed,
                                            minimum_unit="microseconds",
                                        )
                                        + "[/gray78]",
                                        call_id="[gray78]"
                                        + str(event.payload["result"].call.id)[
                                            :8
                                        ]
                                        + "[/gray78]",
                                        total_arguments=len(
                                            event.payload[
                                                "result"
                                            ].call.arguments
                                            or []
                                        ),
                                        result="[spring_green3]"
                                        + event.payload["result"].result
                                        + "[/spring_green3]",
                                    )
                                    if event.type == EventType.TOOL_RESULT
                                    and event.payload["result"]
                                    else (
                                        f"[{precisedelta(event.ellapsed)}]"
                                        f" <{event.type}>: {event.payload}"
                                        if event.payload and event.ellapsed
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
                            )
                        )
                    )  # noqa: E501
                )
                for event in events
                if (
                    (
                        include_tools
                        and event.type in TOOL_TYPES
                        and (
                            event.type != EventType.TOOL_DETECT
                            or include_tool_detect
                        )
                    )
                    or (
                        include_non_tools
                        and event.type not in TOOL_TYPES
                        and (
                            event.type != EventType.TOKEN_GENERATED
                            or include_tokens
                        )
                    )
                )
            ]
        )

        if event_log and events_limit:
            event_log = event_log[-events_limit:]

        return event_log

    def _tokens_table(
        self,
        dbatch: list[Token],
        current_dtoken: Token | None,
        max_dtoken: Token | None,
    ):
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
            is_max_dtoken = max_dtoken and max_dtoken.id == dtoken.id
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
                f"[{dtoken_color}]{_p(dtoken.probability)}[/{dtoken_color}]",
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
                                _f("access_token_name", user.access_token_name)
                                if user
                                else None
                            ),
                        ],
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
    def _symmetric_indices(data: list[float]) -> list[float]:
        """Sorts data desc so that highest values in center lower at edge"""
        assert data
        sorted_data = sorted(data, reverse=True)
        n = len(sorted_data)
        result = [None] * n

        left = n // 2 - 1
        right = n // 2

        for i, value in enumerate(sorted_data):
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
