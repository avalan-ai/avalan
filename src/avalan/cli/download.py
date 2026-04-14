from rich.console import RenderableType
from rich.progress import Progress
from tqdm.std import tqdm

std_tqdm = tqdm


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


class tqdm_rich_progress(tqdm):
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
        progress_options: dict[str, object] = {
            "transient": not self.leave,
            **(raw_progress_options or {}),
        }

        self._progress = Progress(*progress, **progress_options)
        self._progress.__enter__()
        self._task_id = self._progress.add_task(
            self.desc or "", **self.format_dict
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
