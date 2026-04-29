from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict, cast

from rich import filesize
from rich.console import Console, RenderableType
from rich.progress import MofNCompleteColumn, Progress, ProgressColumn, Task
from rich.text import Text
from tqdm.std import tqdm

std_tqdm = tqdm

_INCOMPLETE_TOTAL_DESCRIPTION = "Downloading (incomplete total...)"
_DOWNLOAD_DESCRIPTION = "Downloading"


class _ProgressOptions(TypedDict, total=False):
    auto_refresh: bool
    console: Console | None
    expand: bool
    get_time: Callable[[], float] | None
    redirect_stderr: bool
    redirect_stdout: bool
    refresh_per_second: float
    speed_estimate_period: float
    transient: bool


class _TaskFields(TypedDict, total=False):
    unit: str
    unit_divisor: int
    unit_scale: bool


class _TaskOptions(_TaskFields, total=False):
    completed: int
    total: float | None
    visible: bool


class DownloadCompleteColumn(ProgressColumn):
    """Render byte downloads as sizes and other tasks as completed counts."""

    def __init__(self) -> None:
        super().__init__()
        self._mofn_column = MofNCompleteColumn()

    def render(self, task: Task) -> RenderableType:
        if task.fields.get("unit") == "B":
            return self._render_download(task)
        return self._mofn_column.render(task)

    @staticmethod
    def _render_download(task: Task) -> Text:
        completed = int(task.completed)
        total = int(task.total) if task.total and task.total > 0 else None
        unit_divisor = task.fields.get("unit_divisor", 1000)
        divisor = 1024 if unit_divisor == 1024 else 1000
        suffixes = (
            [
                "bytes",
                "KiB",
                "MiB",
                "GiB",
                "TiB",
                "PiB",
                "EiB",
                "ZiB",
                "YiB",
            ]
            if divisor == 1024
            else [
                "bytes",
                "kB",
                "MB",
                "GB",
                "TB",
                "PB",
                "EB",
                "ZB",
                "YB",
            ]
        )
        unit, suffix = filesize.pick_unit_and_suffix(
            total or completed, suffixes, divisor
        )
        precision = 0 if unit == 1 else 1
        completed_status = f"{completed / unit:,.{precision}f}"
        total_status = f"{total / unit:,.{precision}f}" if total else "?"
        return Text(
            f"{completed_status}/{total_status} {suffix}",
            style="progress.download",
        )


if TYPE_CHECKING:

    class _TqdmBase:
        disable: bool
        leave: bool
        desc: str | None
        n: float
        total: float | None
        format_dict: dict[str, object]

        def __init__(self, *args: object, **kwargs: object) -> None: ...

        def close(self) -> None: ...

        def reset(self, total: int | float | None = None) -> None: ...

else:
    _TqdmBase = tqdm


def create_live_tqdm_class(
    progress_template: tuple[RenderableType, ...],
) -> type["tqdm_rich_progress"]:
    class LiveTqdm(tqdm_rich_progress):
        def __init__(self, *args: object, **kwargs: object) -> None:
            extended_kwargs = {**kwargs}
            extended_kwargs.setdefault("progress", progress_template)
            super().__init__(*args, **extended_kwargs)

    return LiveTqdm


""" Heavily inspired by
    https://github.com/tqdm/tqdm/blob/master/tqdm/rich.py """


class tqdm_rich_progress(_TqdmBase):
    def __init__(self, *args: object, **kwargs: object) -> None:
        sanitized_kwargs = {**kwargs}
        sanitized_kwargs.pop("progress", None)
        sanitized_kwargs.pop("options", None)

        super().__init__(*args, **sanitized_kwargs)
        if self.disable:
            return

        options = {**{"options": None, "progress": None}, **kwargs}

        progress = options.pop("progress")
        assert isinstance(progress, tuple)
        raw_progress_options = options.pop("options", None)
        assert raw_progress_options is None or isinstance(
            raw_progress_options, dict
        )
        progress_options: _ProgressOptions = {"transient": not self.leave}
        if raw_progress_options:
            progress_options.update(
                cast(_ProgressOptions, raw_progress_options)
            )

        self._progress = Progress(*progress, **progress_options)
        self._progress.__enter__()
        task_options = self._task_options()
        self._task_id = self._progress.add_task(
            self._task_description(), **task_options
        )

    def close(self) -> None:
        if self.disable:
            return
        self.display()
        self._progress.__exit__(None, None, None)
        super().close()

    def clear(self, *_: object, **__: object) -> None:
        pass

    def display(self, *_: object, **__: object) -> None:
        if not hasattr(self, "_progress"):
            return
        self._progress.update(
            self._task_id,
            completed=self.n,
            description=self._task_description(),
            total=self.total,
            **self._task_fields(),
        )

    def reset(self, total: int | float | None = None) -> None:
        if hasattr(self, "_progress"):
            self._progress.reset(self._task_id, total=total)
        super().reset(total=total)

    def _task_options(self) -> _TaskOptions:
        task_options: _TaskOptions = {}
        task_total = self.format_dict.get("total")
        if task_total is None:
            task_options["total"] = None
        elif isinstance(task_total, (int, float)):
            task_options["total"] = float(task_total)
        task_completed = self.format_dict.get("n")
        if isinstance(task_completed, (int, float)):
            task_options["completed"] = int(task_completed)
        task_fields = self._task_fields()
        if "unit" in task_fields:
            task_options["unit"] = task_fields["unit"]
        if "unit_divisor" in task_fields:
            task_options["unit_divisor"] = task_fields["unit_divisor"]
        if "unit_scale" in task_fields:
            task_options["unit_scale"] = task_fields["unit_scale"]
        return task_options

    def _task_fields(self) -> _TaskFields:
        task_options: _TaskFields = {}
        task_unit = self.format_dict.get("unit")
        if not isinstance(task_unit, str) or task_unit != "B":
            return task_options
        task_options["unit"] = task_unit
        task_unit_divisor = self.format_dict.get("unit_divisor")
        if isinstance(task_unit_divisor, int):
            task_options["unit_divisor"] = task_unit_divisor
        task_unit_scale = self.format_dict.get("unit_scale")
        if isinstance(task_unit_scale, bool):
            task_options["unit_scale"] = task_unit_scale
        return task_options

    def _task_description(self) -> str:
        if self.desc == _INCOMPLETE_TOTAL_DESCRIPTION:
            return _DOWNLOAD_DESCRIPTION
        return self.desc or ""
