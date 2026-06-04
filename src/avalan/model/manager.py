from ..entities import (
    AttentionImplementation,
    Backend,
    EngineUri,
    Input,
    Modality,
    Operation,
    ParallelStrategy,
    TextGenerationLoaderClass,
    TransformerEngineSettings,
    Vendor,
    WeightType,
)
from ..event import Event, EventType
from ..event.manager import EventManager
from ..secrets import KeyringSecrets
from .call import ModelCall
from .modalities import ModalityRegistry

import asyncio
from argparse import Namespace
from contextlib import AsyncExitStack
from logging import Logger
from os import environ
from time import perf_counter
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Mapping,
    TypeAlias,
    cast,
    get_args,
)
from urllib.parse import parse_qsl, unquote, urlparse

if TYPE_CHECKING:
    from .engine import Engine
else:  # pragma: no cover - runtime type placeholder
    Engine = Any

ModelType: TypeAlias = Engine

_DS4_CONFIG_PREFIX = "ds4_"
_DS4_NATIVE_BACKENDS = frozenset(("auto", "metal", "cuda", "cpu"))
_DS4_CONFIG_KEY_ALIASES = {
    "ctx": "ctx_size",
    "ctx_size": "ctx_size",
    "native_backend": "native_backend",
    "mtp": "mtp_path",
    "mtp_path": "mtp_path",
    "mtp_draft": "mtp_draft_tokens",
    "mtp_draft_tokens": "mtp_draft_tokens",
    "mtp_margin": "mtp_margin",
    "warm_weights": "warm_weights",
    "quality": "quality",
    "native_log": "native_log",
    "directional_steering_file": "directional_steering_file",
    "directional_steering_attn": "directional_steering_attn",
    "directional_steering_ffn": "directional_steering_ffn",
    "kv_disk_dir": "kv_disk_dir",
    "kv_disk_space_mb": "kv_disk_space_mb",
    "seed": "seed",
}
_DS4_NORMALIZED_CONFIG_KEYS = frozenset(_DS4_CONFIG_KEY_ALIASES.values())


def _is_ds4_backend(backend: Backend | str) -> bool:
    return backend == Backend.DS4 or backend == Backend.DS4.value


def _supported_ds4_config_keys() -> str:
    return ", ".join(
        f"{_DS4_CONFIG_PREFIX}{key}" for key in sorted(_DS4_CONFIG_KEY_ALIASES)
    )


def _invalid_ds4_config(key: str, expected: str, value: object) -> ValueError:
    return ValueError(
        f"Invalid DS4 configuration value for {key!r}: expected "
        f"{expected}, got {value!r}."
    )


def _ds4_positive_int(key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _invalid_ds4_config(key, "a positive integer", value)
    return value


def _ds4_non_negative_int(key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _invalid_ds4_config(key, "a non-negative integer", value)
    return value


def _ds4_float(key: str, value: object, *, non_negative: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise _invalid_ds4_config(key, "a number", value)
    float_value = float(value)
    if non_negative and float_value < 0:
        raise _invalid_ds4_config(key, "a non-negative number", value)
    return float_value


def _ds4_string(key: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid_ds4_config(key, "a non-empty string", value)
    return value


def _ds4_bool(key: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise _invalid_ds4_config(key, "a boolean", value)
    return value


def _validate_ds4_config_value(
    key: str, normalized_key: str, value: object
) -> object:
    match normalized_key:
        case "ctx_size":
            return _ds4_positive_int(key, value)
        case "native_backend":
            backend = _ds4_string(key, value).lower()
            if backend not in _DS4_NATIVE_BACKENDS:
                supported = ", ".join(sorted(_DS4_NATIVE_BACKENDS))
                raise _invalid_ds4_config(key, f"one of {supported}", value)
            return backend
        case "mtp_path" | "directional_steering_file" | "kv_disk_dir":
            return _ds4_string(key, value)
        case "mtp_draft_tokens" | "kv_disk_space_mb" | "seed":
            return _ds4_non_negative_int(key, value)
        case "mtp_margin":
            return _ds4_float(key, value, non_negative=True)
        case "directional_steering_attn" | "directional_steering_ffn":
            return _ds4_float(key, value, non_negative=False)
        case "warm_weights" | "quality" | "native_log":
            return _ds4_bool(key, value)
    raise ValueError(
        f"Unknown DS4 backend configuration key {normalized_key!r}."
    )


def _normalize_ds4_backend_config(
    mapping: Mapping[str, object],
    *,
    reject_unknown: bool,
    allow_normalized_keys: bool,
) -> dict[str, object]:
    config: dict[str, object] = {}
    for raw_key, value in mapping.items():
        if value is None:
            continue

        key = raw_key
        if raw_key.startswith(_DS4_CONFIG_PREFIX):
            key = raw_key.removeprefix(_DS4_CONFIG_PREFIX)
        elif allow_normalized_keys and raw_key in _DS4_NORMALIZED_CONFIG_KEYS:
            key = raw_key
        else:
            continue

        normalized_key = _DS4_CONFIG_KEY_ALIASES.get(key)
        if normalized_key is None:
            if reject_unknown:
                supported = _supported_ds4_config_keys()
                raise ValueError(
                    f"Unknown DS4 configuration key {raw_key!r}. "
                    f"Supported DS4 keys: {supported}."
                )
            continue

        config[normalized_key] = _validate_ds4_config_value(
            raw_key, normalized_key, value
        )
    return config


class ModelManager:
    _hub: Any
    _stack: AsyncExitStack
    _logger: Logger
    _secrets: KeyringSecrets
    _event_manager: EventManager | None
    _pending_exit_task: asyncio.Task[None] | None

    def __init__(
        self,
        hub: Any,
        logger: Logger,
        secrets: KeyringSecrets | None = None,
        event_manager: EventManager | None = None,
    ):
        self._hub, self._logger = hub, logger
        self._stack = AsyncExitStack()
        self._secrets = secrets or KeyringSecrets()
        self._event_manager = event_manager
        self._pending_exit_task = None

    def __enter__(self) -> "ModelManager":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._stack.aclose())
        else:
            self._pending_exit_task = loop.create_task(self._stack.aclose())
        return False

    async def __aenter__(self) -> "ModelManager":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._pending_exit_task is not None:
            await self._pending_exit_task
            self._pending_exit_task = None
        return bool(
            await self._stack.__aexit__(exc_type, exc_value, traceback)
        )

    async def __call__(
        self,
        model_task: ModelCall,
    ) -> object:
        modality = model_task.operation.modality

        self._logger.info("ModelManager call process started for %s", modality)

        start = perf_counter()
        if self._event_manager:
            await self._event_manager.trigger(
                Event(
                    type=EventType.MODEL_MANAGER_CALL_BEFORE,
                    payload={
                        "engine_uri": model_task.engine_uri,
                        "modality": modality,
                        "operation": model_task.operation,
                        "context": model_task.context,
                        "task": model_task,
                    },
                    started=start,
                )
            )

        handler = ModalityRegistry.get(modality)
        result = await handler(
            model_task.engine_uri,
            model_task.model,
            model_task.operation,
            model_task.tool,
        )

        end = perf_counter()
        if self._event_manager:
            await self._event_manager.trigger(
                Event(
                    type=EventType.MODEL_MANAGER_CALL_AFTER,
                    payload={
                        "engine_uri": model_task.engine_uri,
                        "modality": modality,
                        "operation": model_task.operation,
                        "context": model_task.context,
                        "task": model_task,
                        "result": result,
                    },
                    started=start,
                    finished=end,
                    elapsed=end - start,
                )
            )

        self._logger.info("ModelManager call processed for %s", modality)

        return result

    @staticmethod
    def ds4_backend_config_from_mapping(
        mapping: Mapping[str, object],
    ) -> dict[str, object]:
        """Return normalized DS4 backend config from prefixed keys."""
        return _normalize_ds4_backend_config(
            mapping,
            reject_unknown=True,
            allow_normalized_keys=True,
        )

    @staticmethod
    def get_operation_from_arguments(
        modality: Modality,
        args: Namespace,
        input_string: Input | None,
    ) -> Operation:
        return ModalityRegistry.get_operation_from_arguments(
            modality, args, input_string
        )

    def get_engine_settings(
        self,
        engine_uri: EngineUri,
        settings: Mapping[str, object] | None = None,
        modality: Modality | None = None,
    ) -> TransformerEngineSettings:
        engine_settings_args: dict[str, Any] = dict(settings or {})
        if "backend" in engine_uri.params:
            backend_param = engine_uri.params["backend"]
            assert isinstance(backend_param, str)
            engine_settings_args["backend"] = Backend(backend_param)
        if "azure_api_version" in engine_uri.params:
            azure_api_version = engine_uri.params["azure_api_version"]
            assert isinstance(azure_api_version, str)
            engine_settings_args["azure_api_version"] = azure_api_version

        backend = engine_settings_args.get("backend", Backend.TRANSFORMERS)
        if _is_ds4_backend(cast(Backend | str, backend)):
            uri_backend_config = _normalize_ds4_backend_config(
                engine_uri.params,
                reject_unknown=True,
                allow_normalized_keys=False,
            )
            explicit_backend_config = _normalize_ds4_backend_config(
                cast(
                    dict[str, object],
                    engine_settings_args.get("backend_config") or {},
                ),
                reject_unknown=True,
                allow_normalized_keys=True,
            )
            ds4_backend_config = {
                **uri_backend_config,
                **explicit_backend_config,
            }
            engine_settings_args["backend_config"] = ds4_backend_config or None

        if modality != Modality.EMBEDDING and not engine_uri.is_local:
            token = None
            if engine_uri.password and engine_uri.user:
                if engine_uri.user == "secret":
                    token = self._secrets.read(engine_uri.password)
                elif engine_uri.user == "env":
                    token = environ.get(engine_uri.password)
                else:
                    token = None
            elif engine_uri.user:
                token = engine_uri.user

            if token:
                engine_settings_args.update(access_token=token)

        engine_settings = TransformerEngineSettings(**engine_settings_args)
        return engine_settings

    def load(
        self,
        engine_uri: EngineUri,
        modality: Modality = Modality.TEXT_GENERATION,
        *args: object,
        attention: AttentionImplementation | None = None,
        base_url: str | None = None,
        device: str | None = None,
        disable_loading_progress_bar: bool = False,
        loader_class: TextGenerationLoaderClass | None = "auto",
        backend: Backend | str = Backend.TRANSFORMERS,
        backend_config: dict[str, object] | None = None,
        low_cpu_mem_usage: bool = False,
        parallel: ParallelStrategy | None = None,
        quiet: bool = False,
        output_hidden_states: bool | None = None,
        base_model_id: str | None = None,
        checkpoint: str | None = None,
        refiner_model_id: str | None = None,
        upsampler_model_id: str | None = None,
        revision: str | None = None,
        special_tokens: list[str] | None = None,
        subfolder: str | None = None,
        tokenizer: str | None = None,
        tokenizer_subfolder: str | None = None,
        tokens: list[str] | None = None,
        trust_remote_code: bool | None = None,
        weight_type: WeightType = "auto",
    ) -> ModelType:
        if "backend" in engine_uri.params:
            backend_param = engine_uri.params["backend"]
            assert isinstance(backend_param, str)
            backend = Backend(backend_param)
        ds4_backend_config: dict[str, object] | None = None
        if _is_ds4_backend(backend):
            uri_backend_config = _normalize_ds4_backend_config(
                engine_uri.params,
                reject_unknown=True,
                allow_normalized_keys=False,
            )
            explicit_backend_config = _normalize_ds4_backend_config(
                backend_config or {},
                reject_unknown=True,
                allow_normalized_keys=True,
            )
            ds4_backend_config = {
                **uri_backend_config,
                **explicit_backend_config,
            }
        engine_settings_args = dict(
            base_url=base_url,
            cache_dir=self._hub.cache_dir,
            device=device,
            disable_loading_progress_bar=quiet or disable_loading_progress_bar,
            low_cpu_mem_usage=low_cpu_mem_usage,
            loader_class=loader_class,
            backend=backend,
            backend_config=ds4_backend_config or None,
            parallel=parallel,
            base_model_id=base_model_id or None,
            checkpoint=checkpoint or None,
            refiner_model_id=refiner_model_id or None,
            upsampler_model_id=upsampler_model_id or None,
            revision=revision,
            special_tokens=special_tokens or None,
            subfolder=subfolder or None,
            tokenizer_name_or_path=tokenizer,
            tokenizer_subfolder=tokenizer_subfolder or None,
            tokens=tokens or None,
            weight_type=weight_type,
        )

        if output_hidden_states is not None:
            engine_settings_args["output_hidden_states"] = output_hidden_states

        if modality != Modality.EMBEDDING:
            engine_settings_args.update(
                attention=attention or None,
                trust_remote_code=trust_remote_code or None,
            )

        engine_settings = self.get_engine_settings(
            engine_uri,
            engine_settings_args,
            modality=modality,
        )
        return self.load_engine(engine_uri, engine_settings, modality)

    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        modality: Modality = Modality.TEXT_GENERATION,
    ) -> ModelType:
        if modality is Modality.EMBEDDING:
            from ..model.nlp.sentence import SentenceTransformerModel

            assert engine_uri.model_id is not None
            model = SentenceTransformerModel(
                model_id=engine_uri.model_id,
                settings=engine_settings,
                logger=self._logger,
            )
        else:
            model = ModalityRegistry.load_engine(
                engine_uri,
                engine_settings,
                modality,
                self._logger,
                self._stack,
            )
        self._stack.enter_context(model)
        return model

    @staticmethod
    def parse_uri(uri: str) -> EngineUri:
        parsed = urlparse(uri)
        if not parsed.scheme:
            uri = f"ai://{uri}"
            parsed = urlparse(uri)

        if parsed.scheme != "ai":
            raise ValueError(
                f"Invalid scheme {parsed.scheme!r}, expected 'ai'"
            )

        vendor = parsed.hostname
        if not vendor or vendor not in get_args(Vendor) or vendor == "local":
            vendor = None
        use_host = bool(vendor)
        path = unquote(parsed.path)
        path_prefixed = path.startswith("/")
        params: dict[str, str | int | float | bool] = {}
        for key, value in parse_qsl(parsed.query):
            if value.lower() in {"true", "false"}:
                params[key] = value.lower() == "true"
            else:
                try:
                    params[key] = int(value)
                except ValueError:
                    try:
                        params[key] = float(value)
                    except ValueError:
                        params[key] = value

        # urlparse() normalizes hostname to lowercase, so keep original case
        authority = parsed.netloc.rsplit("@", 1)[-1]
        hostname = authority.split(":", 1)[0]

        model_id = (
            hostname + ("/" if path_prefixed else "")
            if not vendor and hostname != "local"
            else ""
        ) + (path[1:] if path_prefixed else path)
        engine_uri = EngineUri(
            vendor=cast(Vendor | None, vendor),
            host=hostname if use_host else None,
            port=(parsed.port or None) if use_host else None,
            user=parsed.username or None,
            password=parsed.password or None,
            model_id=model_id,
            params=params,
        )
        return engine_uri
