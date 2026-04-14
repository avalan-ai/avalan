from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict, cast

from rich.console import Console, RenderableType
from rich.progress import Progress
from tqdm.std import tqdm

std_tqdm = tqdm


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


class _TaskOptions(TypedDict, total=False):
    completed: int
    total: float | None
    visible: bool


if TYPE_CHECKING:

    class _TqdmBase:
        disable: bool
        leave: bool
        desc: str | None
        n: float
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
        task_options: _TaskOptions = {}
        task_total = self.format_dict.get("total")
        if task_total is None:
            task_options["total"] = None
        elif isinstance(task_total, (int, float)):
            task_options["total"] = float(task_total)
        task_completed = self.format_dict.get("n")
        if isinstance(task_completed, (int, float)):
            task_options["completed"] = int(task_completed)
        self._task_id = self._progress.add_task(
            self.desc or "", **task_options
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
            self._task_id, completed=self.n, description=self.desc
        )

    def reset(self, total: int | float | None = None) -> None:
        if hasattr(self, "_progress"):
            self._progress.reset(self._task_id, total=total)
        super().reset(total=total)
