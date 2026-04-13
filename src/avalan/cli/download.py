from rich.console import RenderableType
from rich.progress import Progress
from tqdm.std import tqdm as std_tqdm


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


class tqdm_rich_progress(std_tqdm):
    def __init__(self, *args: object, **kwargs: object) -> None:
        sanitized_kwargs = {**kwargs}
        sanitized_kwargs.pop("progress", None)
        sanitized_kwargs.pop("options", None)

        super().__init__(*args, **sanitized_kwargs)  # type: ignore[misc]
        if self.disable:
            return

        options = {**{"options": None, "progress": None}, **kwargs}

        progress = options.pop("progress")
        assert isinstance(progress, tuple)
        progress_options = {
            **{"transient": not self.leave},
            **(options.pop("options", None) or {}),
        }
        assert isinstance(progress_options, dict)

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
            self._progress.reset(total=total)
        super().reset(total=total)
