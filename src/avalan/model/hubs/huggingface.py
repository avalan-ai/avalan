from ... import name, version
from ...entities import (
    Model,
    HubCacheDeletion,
    HubCacheFile,
    HubCache,
    User,
)
from ...model.hubs import HubAccessDeniedException
from datetime import datetime
from huggingface_hub import HfApi, login, ModelInfo, scan_cache_dir
from huggingface_hub.errors import GatedRepoError
from logging import Logger
from os.path import expanduser
from os import getenv
from tqdm import tqdm
from typing import Callable, Iterable, Type
from urllib.parse import urlparse


class HuggingfaceHub:
    DEFAULT_ENDPOINT: str = "https://huggingface.co"
    DEFAULT_CACHE_DIR: str = expanduser(
        getenv("HF_HUB_CACHE") or "~/.cache/huggingface/hub"
    )
    _access_token: str
    _cache_dir: str
    _domain: str
    _hf: HfApi
    _logger: Logger

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    @property
    def domain(self) -> str:
        return self._domain

    def __init__(
        self,
        access_token: str,
        cache_dir: str,
        logger: Logger,
        endpoint: str = DEFAULT_ENDPOINT,
    ) -> None:
        assert access_token and cache_dir
        self._access_token = access_token
        self._hf = HfApi(
            endpoint=endpoint,
            token=access_token,
            library_name=name(),
            library_version=version(),
        )
        self._cache_dir = expanduser(cache_dir)
        self._domain = urlparse(endpoint).netloc
        self._logger = logger

    def cache_delete(
        self, model_id: str, revisions: list[str] | None = None
    ) -> (HubCacheDeletion | None, Callable[[], None] | None):
        scan_results = scan_cache_dir(self._cache_dir)
        delete_revisions = [
            revision.commit_hash
            for info in scan_results.repos
            if info.repo_id == model_id
            for revision in info.revisions
            if not revisions
            or any(revision.commit_hash.startswith(r) for r in revisions)
        ]
        if not delete_revisions:
            return (None, None)

        strategy = scan_results.delete_revisions(*delete_revisions)
        cache_deletion = HubCacheDeletion(
            model_id=model_id,
            revisions=delete_revisions,
            deletable_size_on_disk=strategy.expected_freed_size,
            deletable_blobs=[str(p) for p in strategy.blobs],
            deletable_refs=[str(p) for p in strategy.refs],
            deletable_repos=[str(p) for p in strategy.repos],
            deletable_snapshots=[str(p) for p in strategy.snapshots],
        )
        return cache_deletion, lambda: strategy.execute()

    def cache_scan(
        self, sort_models_by_size: bool = True, sort_files_by_size: bool = True
    ) -> list[HubCache]:
        scan_results = scan_cache_dir(self._cache_dir)
        model_caches = sorted(
            [
                HubCache(
                    model_id=info.repo_id,
                    path=str(info.repo_path),
                    size_on_disk=info.size_on_disk,
                    files={
                        revision.commit_hash: sorted(
                            [
                                HubCacheFile(
                                    name=rfile.file_name,
                                    path=str(rfile.file_path),
                                    size_on_disk=rfile.size_on_disk,
                                    last_accessed=datetime.fromtimestamp(
                                        rfile.blob_last_accessed
                                    ),
                                    last_modified=datetime.fromtimestamp(
                                        rfile.blob_last_modified
                                    ),
                                )
                                for rfile in revision.files
                            ],
                            key=lambda f: f.size_on_disk
                            if sort_files_by_size
                            else f.name,
                            reverse=sort_files_by_size,
                        )
                        for revision in info.revisions
                    },
                    revisions=[r.commit_hash for r in info.revisions],
                    total_files=info.nb_files,
                    total_revisions=len(info.revisions),
                )
                for info in scan_results.repos
            ],
            key=lambda m: m.size_on_disk if sort_models_by_size else m.name,
            reverse=sort_models_by_size,
        )
        return model_caches

    def can_access(self, model_id: str) -> bool:
        try:
            self._hf.auth_check(model_id)
        except GatedRepoError:
            return False
        return True

    def download(
        self,
        model_id: str,
        tqdm_class: Type[tqdm] | Callable[..., tqdm] | None = None,
    ) -> str:
        try:
            path = self._hf.snapshot_download(
                model_id,
                cache_dir=self._cache_dir,
                tqdm_class=tqdm_class,
                force_download=False,
            )
            return path
        except GatedRepoError as e:
            raise HubAccessDeniedException(e)

    def download_all(self, model_id: str) -> list[str]:
        files = self._hf.list_repo_files(model_id)
        for file in files:
            self._hf.hf_hub_download(
                model_id, file, cache_dir=self._cache_dir, force_download=False
            )
        return files

    def model(self, model_id: str) -> Model:
        model_info = self._hf.model_info(model_id)
        return HuggingfaceHub._model(model_info)

    def model_url(self, model_id: str) -> str:
        return f"https://huggingface.co/{model_id}"

    def models(
        self,
        filter: str | None = None,
        name: str | None = None,
        search: str | None = None,
        limit: int | None = None,
    ) -> Iterable[Model]:
        yield from (
            HuggingfaceHub._model(model_info)
            for model_info in self._hf.list_models(
                model_name=name,
                filter=filter,
                search=search,
                limit=limit,
                full=True,
            )
        )

    def login(self) -> None:
        login(self._access_token)

    def user(self) -> User:
        user_result = self._hf.whoami()
        return User(
            name=user_result["name"],
            full_name=user_result["fullname"],
            access_token_name=user_result["auth"]["accessToken"]["displayName"],
        )

    @staticmethod
    def _model(model_info: ModelInfo) -> Model:
        model = Model(
            id=model_info.id,
            parameters=model_info.safetensors.total
            if model_info.safetensors
            else None,
            parameter_types=list(model_info.safetensors.parameters.keys())
            if model_info.safetensors and model_info.safetensors.parameters
            else None,
            inference=model_info.inference,
            library_name=model_info.library_name
            if model_info.library_name
            else model_info.card_data["library_name"]
            if model_info.card_data and "library_name" in model_info.card_data
            else None,
            license=model_info.card_data["license"]
            if model_info.card_data and "license" in model_info.card_data
            else None,
            pipeline_tag=model_info.pipeline_tag,
            tags=model_info.tags,
            architectures=model_info.config["architectures"]
            if model_info.config and "architectures" in model_info.config
            else None,
            model_type=model_info.config["model_type"]
            if model_info.config and "model_type" in model_info.config
            else None,
            auto_model=model_info.transformers_info["auto_model"]
            if model_info.transformers_info
            and "auto_model" in model_info.transformers_info
            else None,
            processor=model_info.transformers_info["processor"]
            if model_info.transformers_info
            and "processor" in model_info.transformers_info
            else None,
            gated=model_info.gated,
            private=model_info.private,
            disabled=model_info.disabled,
            last_downloads=model_info.downloads,
            downloads=model_info.downloads_all_time or model_info.downloads,
            likes=model_info.likes,
            ranking=model_info.trending_score,
            author=model_info.author,
            created_at=model_info.created_at,
            updated_at=model_info.last_modified or model_info.created_at,
        )
        return model
