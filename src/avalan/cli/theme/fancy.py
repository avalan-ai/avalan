from ...agent.orchestrator import Orchestrator
from ...cli.theme import Data, SpinnerName, Theme
from ...entities import (
    EngineMessage,
    HubCache,
    HubCacheDeletion,
    ImageEntity,
    Model,
    ModelConfig,
    SearchMatch,
    SentenceTransformerModelConfig,
    Similarity,
    Token,
    TokenDetail,
    TokenizerConfig,
    ToolCallError,
    ToolCallResult,
    User,
)
from ...event import TOOL_TYPES, Event, EventStats, EventType
from ...memory.partitioner.text import TextPartition
from ...memory.permanent import PermanentMemoryPartition
from ...utils import _j, _lf, to_json

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from locale import format_string
from logging import Logger
from math import ceil, inf
from re import sub
from textwrap import wrap
from typing import Any, Callable, cast
from uuid import UUID

from humanize import (
    clamp,
    intcomma,
    intword,
    naturalday,
    naturalsize,
    precisedelta,
)
from numpy import ndarray
from numpy.linalg import norm
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.spinner import Spinner as RichSpinner
from rich.table import Column, Table
from rich.text import Text


class FancyTheme(Theme):
    """Fancy theme implementation with rich formatting and icons."""

    @property
    def icons(self) -> dict[Data, str]:
        """Return mapping of data keys to emoji icons."""
        return {
            cast(Data, "access_token_name"): ":lock:",
            cast(Data, "agent_id"): ":robot:",
            cast(Data, "agent_output"): ":robot:",
            cast(Data, "avalan"): ":heavy_large_circle:",
            cast(Data, "author"): ":briefcase:",
            cast(Data, "bye"): ":vulcan_salute:",
            cast(Data, "can_access"): ":white_check_mark:",
            cast(Data, "checking_access"): ":mag:",
            cast(Data, "created_at"): ":calendar:",
            cast(Data, "disabled"): ":cross_mark:",
            cast(Data, "download"): ":floppy_disk:",
            cast(Data, "download_access_denied"): ":exclamation_mark:",
            cast(Data, "download_finished"): ":heavy_check_mark:",
            cast(Data, "downloads"): ":floppy_disk:",
            cast(Data, "gated"): ":key:",
            cast(Data, "inference"): ":brain:",
            cast(Data, "input_token_count"): ":laptop_computer:",
            cast(Data, "library_name"): ":books:",
            cast(Data, "license"): ":balance_scale:",
            cast(Data, "likes"): ":orange_heart:",
            cast(Data, "memory"): ":brain:",
            cast(Data, "model_id"): ":name_badge:",
            cast(Data, "model_type"): ":robot_face:",
            cast(Data, "no_access"): ":no_entry_sign:",
            cast(Data, "parameters"): ":abacus:",
            cast(Data, "pipeline_tag"): ":gear:",
            cast(Data, "private"): ":closed_lock_with_key:",
            cast(Data, "ranking"): ":trophy:",
            cast(Data, "path_blobs"): ":file_folder:",
            cast(Data, "path_refs"): ":file_folder:",
            cast(Data, "path_repository"): ":file_folder:",
            cast(Data, "path_snapshot"): ":file_folder:",
            cast(Data, "session"): ":card_index_dividers:",
            cast(Data, "task_id"): ":robot:",
            cast(Data, "total_tokens"): ":abacus:",
            cast(Data, "tokens_rate"): ":high_voltage:",
            cast(Data, "events"): ":bookmark_tabs:",
            cast(Data, "tool_calls"): ":hammer:",
            cast(Data, "tool_call_results"): ":package:",
            cast(Data, "ttft"): ":seedling:",
            cast(Data, "ttnt"): ":alarm_clock:",
            cast(Data, "ttsr"): ":thinking_face:",
            cast(Data, "updated_at"): ":calendar:",
            cast(Data, "user"): ":hugging_face:",
            cast(Data, "user_input"): ":speaking_head:",
            cast(Data, "tags"): ":label:",
        }

    @property
    def styles(self) -> dict[Data, str]:
        """Return mapping of data keys to rich styles."""
        return {
            cast(Data, "id"): "bold",
            cast(Data, "can_access"): "green",
            cast(Data, "checking_access"): "bright_black blink",
            cast(Data, "created_at"): "magenta",
            cast(Data, "downloads"): "bright_black",
            cast(Data, "likes"): "bright_black",
            cast(Data, "memory"): "magenta",
            cast(Data, "memory_embedding_comparison"): "dark_orange3",
            cast(
                Data, "memory_embedding_comparison_similarity"
            ): "dark_orange3",
            cast(
                Data, "memory_embedding_comparison_similarity_high"
            ): "bold dark_olive_green3",
            cast(
                Data, "memory_embedding_comparison_similarity_middle"
            ): "orange_red1",
            cast(
                Data, "memory_embedding_comparison_similarity_low"
            ): "dark_red",
            cast(Data, "model_id"): "cyan",
            cast(Data, "no_access"): "bold red",
            cast(Data, "parameters"): "bold cyan",
            cast(Data, "participant_id"): "bold",
            cast(Data, "ranking"): "bright_black",
            cast(Data, "session_id"): "dark_orange3",
            cast(Data, "score"): "dark_orange3",
            cast(Data, "tags"): "gray30",
            cast(Data, "updated_at"): "magenta",
            cast(Data, "user"): "bold cyan",
            cast(Data, "version"): "bold",
        }

    @property
    def spinners(self) -> dict[SpinnerName, str]:
        """Return mapping of spinner names to rich spinner types."""
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
        """Return list of data keys that should use quantity formatting."""
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
        """Render an action panel."""
        _i = self._icons
        description_color = (
            "green" if finished else "white" if highlight else "gray62"
        )
        author_icon = _i.get(cast(Data, "author")) or ""
        library_icon = _i.get(cast(Data, "library_name")) or ""
        task_icon = _i.get(cast(Data, "task_id")) or ""
        model_icon = _i.get(cast(Data, "model_id")) or ""
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
                                author_icon
                                + (
                                    f" [bright_black]{author}[/bright_black]"
                                    + " · "
                                    + library_icon
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
            title=(task_icon + f" [cyan]{name}[/cyan]"),
            subtitle=(
                model_icon + f" [bright_black]{model_id}[/bright_black]"
            ),
            box=box.DOUBLE if highlight else box.SQUARE,
        )

    def agent(
        self,
        agent: Orchestrator,
        *args: Any,
        models: list[Model | str],
        can_access: bool | None,
    ) -> RenderableType:
        """Render an agent panel with model information."""
        _, _f, _i = self._, self._f, self._icons
        model_id_icon = _i.get(cast(Data, "model_id")) or ""
        models_group = Group(
            *_lf(
                [
                    model_id_icon
                    + " "
                    + ", ".join(
                        [
                            (
                                _("{model_id} ({parameters})").format(
                                    model_id=_f(
                                        cast(Data, "model_id"),
                                        model.id,
                                        icon=False,
                                    ),
                                    parameters=_f(
                                        cast(Data, "parameters"),
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
                        cast(Data, "memory"),
                        _j(
                            ", ",
                            cast(
                                list[str],
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
                                ],
                            ),
                            empty=_("stateless"),
                        ),
                    ),
                    (
                        _f(
                            cast(Data, "session"),
                            " "
                            + _("session: {session_id}").format(
                                session_id=_f(
                                    cast(Data, "session_id"),
                                    (
                                        str(
                                            agent.memory.permanent_message.session_id
                                        )
                                        if agent.memory.permanent_message
                                        else ""
                                    ),
                                )
                            ),
                        )
                        if agent.memory.has_permanent_message
                        and agent.memory.permanent_message
                        and agent.memory.permanent_message.has_session
                        else None
                    ),
                ]
            )
        )
        return Panel(
            models_group,
            title=_f(
                cast(Data, "agent_id"),
                agent.name if agent.name else str(agent.id),
            ),
            box=box.DOUBLE,
        )

    def ask_access_token(self) -> str:
        """Return prompt text for access token input."""
        _ = self._
        return _("Enter your Huggingface access token")

    def ask_delete_paths(self) -> str:
        """Return prompt text for delete paths confirmation."""
        _ = self._
        return _("Delete selected paths?")

    def ask_login_to_hub(self) -> str:
        """Return prompt text for hub login confirmation."""
        _ = self._
        return _("Login to huggingface?")

    def ask_secret_password(self, key: str) -> str:
        """Return prompt text for secret password input."""
        _ = self._
        return _("Enter secret for {key}").format(key=key)

    def ask_override_secret(self, key: str) -> str:
        """Return prompt text for secret override confirmation."""
        _ = self._
        return _("Secret {key} exists, override?").format(key=key)

    def bye(self) -> RenderableType:
        """Return goodbye message."""
        _, _i = self._, self._icons
        bye_icon = _i.get(cast(Data, "bye")) or ""
        return bye_icon + " " + _("bye :)")

    def cache_delete(
        self, cache_deletion: HubCacheDeletion | None, deleted: bool = False
    ) -> RenderableType:
        """Render cache deletion summary."""
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
                    model_id=_f(
                        cast(Data, "model_id"), cache_deletion.model_id
                    ),
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
                            *[
                                _f(cast(Data, field_name), path)
                                for path in deletable_paths
                            ]
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
                    model_id=_f(
                        cast(Data, "model_id"), cache_deletion.model_id
                    ),
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
        """Render cache list table."""
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
            filtered_models: list[HubCache] = (
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
        """Render download access denied message."""
        _, _i = self._, self._icons
        access_denied_icon = _i.get(cast(Data, "download_access_denied")) or ""
        return Group(
            *_lf(
                [
                    Padding(
                        " ".join(
                            [
                                "[bold red]"
                                + access_denied_icon
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
        """Render download start message."""
        _, _i = self._, self._icons
        download_icon = _i.get(cast(Data, "download")) or ""
        return Group(
            download_icon
            + " "
            + _("Downloading model {model_id}:").format(model_id=model_id),
            "",
        )

    def download_progress(
        self,
    ) -> tuple[str | RenderableType]:  # type: ignore[override]
        """Return progress bar components for download."""
        return (  # type: ignore[return-value]
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
        """Render download finished message."""
        _, _i = self._, self._icons
        finished_icon = _i.get(cast(Data, "download_finished")) or ""
        return Padding(
            " ".join(
                [
                    "[bold green]" + finished_icon + "[/bold green]",
                    _("Downloaded model {model_id} to {path}").format(
                        model_id=model_id, path=path
                    ),
                ]
            )
        )

    def events(  # type: ignore[override]
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
        """Render events panel."""
        _ = self._

        event_log = self._events_log(
            events=events,
            events_limit=events_limit,
            include_tokens=include_tokens,
            include_tool_detect=include_tool_detect,
            include_tools=include_tools,
            include_non_tools=include_non_tools,
        )
        panel: Panel | None = (
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
        """Return logging in message."""
        _ = self._
        return _("Logging in to {domain}...").format(domain=domain)

    def memory_embeddings(
        self,
        input_string: str,
        embeddings: ndarray,
        *args: Any,
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
        """Render memory embeddings table."""
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

                columns: list[str] = []
                for v in embeddings[:embedding_peek]:
                    columns.append(clamp(v, format="{:.4g}"))

                columns.append("")

                for v in embeddings[-embedding_peek:]:
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
        """Render memory embeddings comparison table."""
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
                    cast(Data, field_class),
                    compare_string,
                    icon=":trophy: " if is_most else False,
                ),
                _f(
                    cast(Data, field_class),
                    clamp(similarity.cosine_distance, format="{:.4g}"),
                    icon=False,
                ),
                _f(
                    cast(Data, field_class),
                    clamp(similarity.l1_distance, format="{:.4g}"),
                    icon=False,
                ),
                _f(
                    cast(Data, field_class),
                    clamp(similarity.l2_distance, format="{:.4g}"),
                    icon=False,
                ),
                _f(
                    cast(Data, field_class),
                    clamp(similarity.inner_product, format="{:.4g}"),
                    icon=False,
                ),
                _f(
                    cast(Data, field_class),
                    clamp(similarity.pearson, format="{:.4g}"),
                    icon=False,
                ),
            )
        return Align(table, align="center")

    def memory_embeddings_search(
        self,
        matches: list[SearchMatch],
        *args: Any,
        match_preview_length: int = 300,
    ) -> RenderableType:
        """Render memory embeddings search results."""
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
                    cast(Data, field_class),
                    match.query,
                    icon=":trophy: " if is_most else False,
                ),
                _f(
                    cast(Data, field_class),
                    (
                        match.match
                        if len(match.match) <= match_preview_length
                        else match.match[:match_preview_length] + "..."
                    ),
                    icon=False,
                ),
                _f(
                    cast(Data, field_class),
                    clamp(match.l2_distance, format="{:.4g}"),
                    icon=False,
                ),
            )
        return Align(table, align="center")

    def memory_partitions(
        self,
        partitions: list[TextPartition],
        *args: Any,
        display_partitions: int,
    ) -> RenderableType:
        """Render memory partitions."""
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
                Align(Padding(self._("..."), pad=(0, 0, 1, 0)), align="center")
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
        *args: Any,
        can_access: bool | None = None,
        expand: bool = False,
        summary: bool = False,
    ) -> RenderableType:
        """Render model panel."""
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
                            cast(
                                list[str],
                                [
                                    _j(
                                        " ",
                                        cast(
                                            list[str],
                                            [
                                                (
                                                    _f(
                                                        cast(
                                                            Data,
                                                            "checking_access",
                                                        ),
                                                        _("checking access"),
                                                    )
                                                    if can_access is None
                                                    else (
                                                        _f(
                                                            cast(
                                                                Data,
                                                                "can_access",
                                                            ),
                                                            _(
                                                                "access"
                                                                " granted"
                                                            ),
                                                        )
                                                        if can_access
                                                        else _f(
                                                            cast(
                                                                Data,
                                                                "no_access",
                                                            ),
                                                            _("access denied"),
                                                        )
                                                    )
                                                ),
                                                _f(
                                                    cast(Data, "author"),
                                                    model.author,
                                                ),
                                                (
                                                    _f(
                                                        cast(Data, "license"),
                                                        model.license,
                                                    )
                                                    if expand and model.license
                                                    else None
                                                ),
                                                (
                                                    _f(
                                                        cast(Data, "gated"),
                                                        _("gated"),
                                                    )
                                                    if model.gated
                                                    else None
                                                ),
                                                (
                                                    _f(
                                                        cast(Data, "private"),
                                                        _("private"),
                                                    )
                                                    if model.private
                                                    else None
                                                ),
                                                (
                                                    _f(
                                                        cast(Data, "disabled"),
                                                        _("disabled"),
                                                    )
                                                    if model.disabled
                                                    else None
                                                ),
                                            ],
                                        ),
                                    ),
                                    (
                                        (
                                            (
                                                _i.get(
                                                    cast(Data, "created_at")
                                                )
                                                or ""
                                            )
                                            + " "
                                            + _j(
                                                ", ",
                                                cast(
                                                    list[str],
                                                    [
                                                        (
                                                            _f(
                                                                cast(
                                                                    Data,
                                                                    "created_at",
                                                                ),
                                                                model.created_at,
                                                                _("created: "),
                                                                icon=False,
                                                            )
                                                            if expand
                                                            else None
                                                        ),
                                                        _f(
                                                            cast(
                                                                Data,
                                                                "updated_at",
                                                            ),
                                                            model.updated_at,
                                                            _("updated: "),
                                                            icon=False,
                                                        ),
                                                    ],
                                                ),
                                            )
                                        )
                                        if not summary
                                        else None
                                    ),
                                ],
                            ),
                        ),
                        (
                            _j(
                                " · ",
                                cast(
                                    list[str],
                                    [
                                        (
                                            _f(
                                                cast(Data, "model_type"),
                                                model.model_type,
                                            )
                                            + (
                                                " ("
                                                + ", ".join(
                                                    model.architectures
                                                )
                                                + ")"
                                                if expand
                                                and model.architectures
                                                else ""
                                            )
                                            if model.model_type
                                            else None
                                        ),
                                        (
                                            _f(
                                                cast(Data, "library_name"),
                                                model.library_name,
                                            )
                                            if model.library_name
                                            else None
                                        ),
                                        (
                                            _f(
                                                cast(Data, "inference"),
                                                model.inference,
                                            )
                                            if expand and model.inference
                                            else None
                                        ),
                                        (
                                            _f(
                                                cast(Data, "pipeline_tag"),
                                                model.pipeline_tag,
                                            )
                                            if model.pipeline_tag
                                            else None
                                        ),
                                    ],
                                ),
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
                            _f(cast(Data, "tags"), " " + ", ".join(model.tags))
                            if expand and model.tags
                            else None
                        ),
                    ]
                )
            ),
            # Model ID
            title=(
                _f(cast(Data, "model_id"), model.id)
                + (
                    " "
                    + _j(
                        " ",
                        cast(
                            list[str],
                            [
                                _f(
                                    cast(Data, "parameters"),
                                    self._parameter_count(model.parameters),
                                ),
                                (
                                    _f(
                                        cast(Data, "parameter_types"),
                                        ", ".join(model.parameter_types),
                                    )
                                    if expand and model.parameter_types
                                    else None
                                ),
                                _("parameters") if expand else _("params"),
                            ],
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
                    cast(
                        list[str],
                        [
                            (
                                _f(cast(Data, "downloads"), model.downloads)
                                if model.downloads
                                else None
                            ),
                            (
                                _f(cast(Data, "likes"), model.likes)
                                if model.likes
                                else None
                            ),
                            (
                                _f(cast(Data, "ranking"), model.ranking)
                                if model.ranking
                                else None
                            ),
                        ],
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
        tokenizer_config: TokenizerConfig,
        *args: Any,
        is_runnable: bool | None = None,
        summary: bool = False,
    ) -> RenderableType:
        """Render model display with config and tokenizer info."""
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
        *args: Any,
        is_runnable: bool | None,
        summary: bool,
    ) -> RenderableType:
        """Render sentence transformer model config table."""
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
        *args: Any,
        is_runnable: bool | None,
        summary: bool,
    ) -> RenderableType:
        """Render model config table."""
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
        *args: Any,
        is_runnable: bool | None,
        summary: bool,
    ) -> Table:
        """Fill model config table with configuration details."""
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
                f"[magenta]{intcomma(config.vocab_size or 0)}[/magenta]",
            )
        config_table.add_row(
            _("Hidden size"),
            f"[magenta]{intcomma(config.hidden_size or 0)}[/magenta]",
        )
        if not summary:
            num_hidden = config.num_hidden_layers or 0
            config_table.add_row(
                _("Number of hidden layers"),
                f"[magenta]{intcomma(num_hidden)}[/magenta]",
            )
            config_table.add_row(
                _("Number of attention heads"),
                f"[magenta]{intcomma(config.num_attention_heads or 0)}"
                "[/magenta]",
            )
            config_table.add_row(
                _("Number of labels in last layer"),
                f"[magenta]{intcomma(config.num_labels or 0)}[/magenta]",
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
        """Render recent messages panel."""
        _, _f, _i = self._, self._f, self._icons
        agent_output_icon = _i.get(cast(Data, "agent_output")) or ""
        user_input_icon = _i.get(cast(Data, "user_input")) or ""
        group = Group(
            *_lf(
                [
                    Panel(
                        (
                            str(engine_message.message.content)
                            if engine_message.message.content
                            else ""
                        ),
                        title=(
                            agent_output_icon
                            + " "
                            + _f(cast(Data, "id"), agent.name)
                            if engine_message.is_from_agent
                            else user_input_icon
                            + "  "
                            + _f(
                                cast(Data, "participant_id"),
                                str(participant_id),
                            )
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

    def saved_tokenizer_files(  # type: ignore[override]
        self, directory_path: str, total_files: int
    ) -> RenderableType:
        """Render saved tokenizer files message."""
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
        messages: list[EngineMessage],
    ) -> RenderableType:
        """Render search message matches panel."""
        _, _f, _i = self._, self._f, self._icons
        agent_output_icon = _i.get(cast(Data, "agent_output")) or ""
        user_input_icon = _i.get(cast(Data, "user_input")) or ""
        group = Group(
            *_lf(
                [
                    Panel(
                        (
                            str(engine_message.message.content)
                            if engine_message.message.content
                            else ""
                        ),
                        title=(
                            agent_output_icon
                            + " "
                            + _f(
                                cast(Data, "id"),
                                agent.name or str(agent.id),
                            )
                            if engine_message.is_from_agent
                            else user_input_icon
                            + "  "
                            + _f(
                                cast(Data, "participant_id"),
                                str(participant_id),
                            )
                        ),
                        title_align="left",
                        subtitle=_("Matching score: {score}").format(
                            score=_f(
                                cast(Data, "score"),
                                clamp(
                                    getattr(engine_message, "score", 0.0),
                                    format="{:.8g}",
                                ),
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
        """Render memory search matches panel."""
        _, _f, _i = self._, self._f, self._icons
        memory_icon = _i.get(cast(Data, "memory")) or ""
        group = Group(
            *_lf(
                [
                    Panel(
                        memory.data,
                        title=(
                            memory_icon
                            + " "
                            + _f(cast(Data, "id"), str(memory.memory_id))
                        ),
                        title_align="left",
                        subtitle=_(
                            "Participant: {participant} – Namespace: {ns} –"
                            " Partition: {partition}"
                        ).format(
                            participant=_f(
                                cast(Data, "participant_id"),
                                str(participant_id),
                            ),
                            ns=_f(cast(Data, "id"), namespace),
                            partition=_f(
                                cast(Data, "number"), memory.partition
                            ),
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
        self, config: TokenizerConfig, *args: Any, summary: bool = False
    ) -> RenderableType:
        """Render tokenizer config table."""
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
            if config.special_tokens:
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

    def tokenizer_tokens(  # type: ignore[override]
        self,
        dtokens: list[Token],
        added_tokens: list[str] | None,
        special_tokens: list[str] | None,
        display_details: bool = False,
        current_dtoken: Token | None = None,
        dtokens_selected: list[Token] | None = None,
    ) -> RenderableType:
        """Render tokenizer tokens panel."""
        if dtokens_selected is None:
            dtokens_selected = []
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

    def display_image_entities(
        self, entities: list[ImageEntity], sort: bool
    ) -> RenderableType:
        """Render image entities table."""
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
                self._f(cast(Data, "score"), f"{entity.score:.2f}")
                if entity.score is not None
                else "-"
            )
            entity_box = (
                ", ".join(f"{v:.2f}" for v in entity.box)
                if entity.box
                else "-"
            )
            table.add_row(entity.label, score, entity_box)

        return Align(table, align="center")

    def display_image_entity(self, entity: ImageEntity) -> RenderableType:
        """Render single image entity table."""
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
        """Render audio labels table."""
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
            score_text = self._f(cast(Data, "score"), f"{score:.2f}")
            table.add_row(label, score_text)
        return Align(table, align="center")

    def display_image_labels(self, labels: list[str]) -> RenderableType:
        """Render image labels table."""
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
        """Render token labels table."""
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
        tool_running_spinner: RichSpinner | None,
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
    ) -> AsyncGenerator[tuple[Token | None, RenderableType], None]:
        """Generate token display panels asynchronously."""
        _, _n, _f, _l = self._, self._n, self._f, logger.debug

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

        dtokens: list[Token] | None = (
            tokens[-display_token_size:]
            if display_token_size and tokens
            else None
        )
        dtokens_selected: list[TokenDetail] | None = (
            [
                dtoken
                for dtoken in dtokens
                if isinstance(dtoken, TokenDetail)
                and focus_on_token_when
                and focus_on_token_when(dtoken)
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
                        cast(Data, "input_token_count"),
                        _n(
                            "{total_tokens} token in",
                            "{total_tokens} tokens in",
                            input_token_count,
                        ).format(total_tokens=input_token_count),
                    ),
                    _f(
                        cast(Data, "total_tokens"),
                        _n(
                            "{total_tokens} token out",
                            "{total_tokens} tokens out",
                            total_tokens,
                        ).format(total_tokens=total_tokens),
                    ),
                    (
                        _f(
                            cast(Data, "ttft"),
                            _("ttft: {ttft} s").format(ttft=f"{ttft:.2f}"),
                        )
                        if ttft
                        else None
                    ),
                    (
                        _f(
                            cast(Data, "ttnt"),
                            _("ttnt: {ttnt} s").format(ttnt=f"{ttnt:.1f}"),
                        )
                        if ttnt
                        else None
                    ),
                    (
                        _f(
                            cast(Data, "ttsr"),
                            _("rt: {ttsr} s").format(ttsr=f"{ttsr:.1f}"),
                        )
                        if ttsr
                        else None
                    ),
                    _f(
                        cast(Data, "tokens_rate"),
                        _("{tokens_rate} t/s").format(
                            tokens_rate=f"{total_tokens / elapsed:.2f}"
                        ),
                    ),
                    (
                        _f(
                            cast(Data, "events"),
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
                            cast(Data, "tool_calls"),
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
                            cast(Data, "tool_call_results"),
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
                    model_id=_f(cast(Data, "id"), model_id)
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
                        model_id=_f(cast(Data, "id"), model_id)
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

        if (
            tool_running_spinner
            and tool_event_calls is not None
            and tool_event_results is not None
            and len(tool_event_calls) != len(tool_event_results)
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
                            think_panel or None,
                            tool_panel or None,
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
                current_data: list[float | None] | None = None
                current_dtoken: TokenDetail | None = None
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
                                current_dtoken.probability or 0.0,
                                format="{:.4g}",
                            )
                            + f"and {current_dtoken.tokens}"
                        )

                tokens_panel = self.tokenizer_tokens(
                    dtokens if dtokens else [],
                    added_tokens,
                    special_tokens,
                    display_details=False,
                    current_dtoken=current_dtoken,
                    dtokens_selected=cast(
                        list[Token] | None, dtokens_selected
                    ),
                )

                # Build bar chart with token alternative probabilities
                chart = None
                if display_probabilities:
                    current_symmetric_indices: list[int] | None = (
                        FancyTheme._symmetric_indices(
                            [v or 0.0 for v in current_data]
                        )
                        if current_data
                        else None
                    )
                    current_symmetric_data: list[float] | None = (
                        [
                            (current_data[i] or 0.0)
                            for i in current_symmetric_indices
                        ]
                        if current_data and current_symmetric_indices
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
                            0.0 for _ in range(pick)
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
                        dtoken_tokens,
                        key=lambda dt: dt.probability or 0.0,
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
                            think_panel or None,
                            tool_panel or None,
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
        """Generate event log entries."""
        _, _n = self._, self._n
        if not events or events_limit == 0:
            return None

        event_log: list[str] | None = _lf(
            [
                self._format_event(event, _, _n)
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

    def _format_event(
        self,
        event: Event,
        _: Callable[[str], str],
        _n: Callable[[str, str, int], str],
    ) -> str | None:
        """Format a single event for display."""
        payload = event.payload

        if event.type == EventType.TOOL_EXECUTE and payload:
            call = payload.get("call")
            if call:
                return _(
                    "Executing tool {tool} call #{call_id} with"
                    " {total_arguments} arguments: {arguments}."
                ).format(
                    tool="[gray78]" + call.name + "[/gray78]",
                    call_id="[gray78]" + str(call.id)[:8] + "[/gray78]",
                    total_arguments=len(call.arguments or []),
                    arguments="[gray78]"
                    + (
                        s
                        if len(s := str(call.arguments)) <= 50
                        else s[:47] + "..."
                    )
                    + "[/gray78]",
                )

        if event.type == EventType.TOOL_MODEL_RUN and payload:
            messages = payload.get("messages", [])
            model_id = payload.get("model_id", "")
            return _n(
                "Running ReACT model {model_id} with {total_messages} message",
                "Running ReACT model {model_id} with {total_messages}"
                " messages",
                len(messages),
            ).format(
                model_id=model_id,
                total_messages=len(messages),
            )

        if event.type == EventType.TOOL_MODEL_RESPONSE and payload:
            model_id = payload.get("model_id", "")
            return _("Got ReACT response from model {model_id}").format(
                model_id=model_id
            )

        if event.type == EventType.TOOL_PROCESS and payload:
            calls: list[Any] = payload if isinstance(payload, list) else []
            return _n(
                "Executing {total_calls} tool: {calls}",
                "Executing {total_calls} tools: {calls}",
                len(calls),
            ).format(
                total_calls=len(calls),
                calls="[gray78]"
                + "[/gray78], [gray78]".join(
                    [c.name for c in calls if hasattr(c, "name")]
                )
                + "[/gray78]",
            )

        if event.type == EventType.TOOL_RESULT and payload:
            result = payload.get("result")
            if result:
                call = getattr(result, "call", None)
                if call:
                    elapsed_delta = (
                        timedelta(seconds=event.elapsed)
                        if event.elapsed is not None
                        else timedelta(seconds=0)
                    )
                    result_text: str
                    if isinstance(result, ToolCallError):
                        result_text = "[red]" + result.message + "[/red]"
                    elif isinstance(result, ToolCallResult):
                        result_text = (
                            "[spring_green3]"
                            + to_json(result.result)
                            + "[/spring_green3]"
                        )
                    else:
                        result_text = str(result)
                    return _(
                        "Executed tool {tool} call #{call_id} with"
                        ' {total_arguments} arguments. Got result "{result}"'
                        " in {elapsed_with_unit}."
                    ).format(
                        tool="[gray78]" + call.name + "[/gray78]",
                        elapsed_with_unit="[gray78]"
                        + precisedelta(
                            elapsed_delta,
                            minimum_unit="microseconds",
                        )
                        + "[/gray78]",
                        call_id="[gray78]" + str(call.id)[:8] + "[/gray78]",
                        total_arguments=len(call.arguments or []),
                        result=result_text,
                    )

        # Default format for other event types
        if payload and event.elapsed is not None:
            elapsed_delta = timedelta(seconds=event.elapsed)
            return f"[{precisedelta(elapsed_delta)}] <{event.type}>: {payload}"
        if payload and event.started:
            return (
                f"[{datetime.fromtimestamp(event.started).isoformat(sep=' ', timespec='seconds')}]"  # noqa: E501
                f" <{event.type}>: {payload}"
            )
        if payload:
            return (
                f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}]"
                f" <{event.type}>: {payload}"
            )
        return (
            f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}]"
            f" <{event.type}>"
        )

    def _tokens_table(
        self,
        dbatch: list[Token],
        current_dtoken: Token | None,
        max_dtoken: Token | None,
    ) -> Table:
        """Build token alternatives table."""
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
        """Render welcome panel."""
        _, _f, _i = self._, self._f, self._icons
        avalan_icon = _i.get(cast(Data, "avalan")) or ""
        license_icon = _i.get(cast(Data, "license")) or ""
        license_text = _("{license} license").format(license=license)
        return Padding(
            Panel(
                Padding(
                    _j(
                        " - ",
                        cast(
                            list[str],
                            [
                                " ".join(
                                    [
                                        avalan_icon
                                        + f" [link={url}]{name}[/link]",
                                        f"[version]{version}[/version]",
                                        "[bright_black]"
                                        + license_icon
                                        + f" {license_text}[/bright_black]",
                                    ]
                                ),
                                (
                                    _f(cast(Data, "user"), user.name)
                                    if user
                                    else None
                                ),
                                (
                                    _f(
                                        cast(Data, "access_token_name"),
                                        user.access_token_name,
                                    )
                                    if user
                                    else None
                                ),
                            ],
                        ),
                    )
                ),
                box=box.SQUARE,
            ),
            pad=(0, 0, 0, 0),  # Might bring lower padding back (0,0,1,0)
        )

    def _parameter_count(self, parameters: int | None) -> str:
        """Format parameter count for display."""
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
        """Sort data desc so that highest values in center lower at edge."""
        assert data
        sorted_data = sorted(data, reverse=True)
        n = len(sorted_data)
        result: list[int | None] = [None] * n

        left = n // 2 - 1
        right = n // 2

        for i, _ in enumerate(sorted_data):
            if i % 2 == 0:
                result[left] = i
                left -= 1
            else:
                result[right] = i
                right += 1
        return [r for r in result if r is not None]

    @staticmethod
    def _percentage(value: float) -> str:
        """Format value as percentage."""
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
        """Wrap text tokens to specified width."""
        lines: list[str] = []
        output = "".join(text_tokens)
        for line in output.splitlines():
            wrapped_line = wrap(line, width=width)
            if wrapped_line:
                lines.extend(wrapped_line)
            elif not skip_blank_lines:
                lines.append("")
        return lines
