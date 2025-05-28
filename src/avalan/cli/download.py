from rich.console import RenderableType
from rich.progress import Progress
from tqdm.std import tqdm as std_tqdm
from typing import Tuple


def create_live_tqdm_class(progress_template: Tuple[RenderableType, ...]):
    class LiveTqdm(tqdm_rich_progress):
        def __init__(self, *args, **kwargs):
            extended_kwargs = {**kwargs}
            extended_kwargs.setdefault("progress", progress_template)
            super().__init__(*args, **extended_kwargs)

    return LiveTqdm


""" Heavily inspired by https://github.com/tqdm/tqdm/blob/master/tqdm/rich.py """


class tqdm_rich_progress(std_tqdm):
    def __init__(self, *args, **kwargs):
        sanitized_kwargs = {**kwargs}
        sanitized_kwargs.pop("progress", None)
        sanitized_kwargs.pop("options", None)

        super().__init__(*args, **sanitized_kwargs)
        if self.disable:
            return

        options = {**{"options": None, "progress": None}, **kwargs}

        progress = options.pop("progress")
        progress_options = {
            **{"transient": not self.leave},
            **(options.pop("options", None) or {}),
        }

        self._progress = Progress(*progress, **progress_options)
        self._progress.__enter__()
        self._task_id = self._progress.add_task(
            self.desc or "", **self.format_dict
        )

    def close(self):
        if self.disable:
            return
        self.display()
        self._progress.__exit__(None, None, None)
        super().close()

    def clear(self, *_, **__):
        pass

    def display(self, *_, **__):
        if not hasattr(self, "_progress"):
            return
        self._progress.update(
            self._task_id, completed=self.n, description=self.desc
        )

    def reset(self, total=None):
        if hasattr(self, "_progress"):
            self._progress.reset(total=total)
        super().reset(total=total)
