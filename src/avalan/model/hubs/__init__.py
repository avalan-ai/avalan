from ...entities import HubCache, HubCacheDeletion, Model

from collections.abc import Callable, Iterable
from typing import Protocol


class HubAccessDeniedException(Exception):
    pass


class HubClient(Protocol):
    @property
    def cache_dir(self) -> str: ...

    def cache_delete(
        self, model_id: str, revisions: list[str] | None = None
    ) -> tuple[HubCacheDeletion | None, Callable[[], None] | None]: ...

    def cache_scan(
        self, sort_models_by_size: bool = True, sort_files_by_size: bool = True
    ) -> list[HubCache]: ...

    def can_access(self, model_id: str) -> bool: ...

    def download(
        self,
        model_id: str,
        *,
        workers: int = 8,
        tqdm_class: type[object] | None = None,
        local_dir: str | None = None,
        local_dir_use_symlinks: bool | None = None,
    ) -> str: ...

    def model(self, model_id: str) -> Model: ...

    def model_url(self, model_id: str) -> str: ...

    def models(
        self,
        *,
        filter: str | None = None,
        search: str | None = None,
        author: str | None = None,
        gated: bool | None = None,
        language: str | None = None,
        library: str | None = None,
        name: str | None = None,
        task: str | None = None,
        tags: list[str] | None = None,
        trained_dataset: str | None = None,
        limit: int = 10,
    ) -> Iterable[Model]: ...
