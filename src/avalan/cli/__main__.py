from .. import license, name, site, version
from ..cli import CommandAbortException, has_input
from ..cli.commands import is_ds4_backend_selected
from ..cli.theme_registry import (
    DEFAULT_THEME_NAME,
    SUPPORTED_THEME_NAMES,
    create_theme,
)
from ..entities import (
    AttentionImplementation,
    Backend,
    BetaSchedule,
    DistanceType,
    GenerationCacheStrategy,
    Modality,
    ParallelStrategy,
    ReasoningEffort,
    ReasoningTag,
    TextGenerationLoaderClass,
    TimestepSpacing,
    ToolCallRecoveryFormat,
    ToolFormat,
    ToolNamePolicyMode,
    User,
    VisionColorModel,
    VisionImageFormat,
    WeightType,
)
from ..filesystem import read_text
from ..model.manager import ModelManager
from ..server_output_redaction import (
    SERVER_OUTPUT_REDACTION_CHANNELS,
    SERVER_OUTPUT_REDACTION_PROTOCOLS,
    SERVER_OUTPUT_REDACTION_RULES,
)
from ..skill import SkillSourceAuthorityKind
from ..tool.browser import BrowserToolSettings
from ..tool.database.settings import DatabaseToolSettings
from ..tool.graph_settings import GraphToolSettings
from ..tool.shell import ShellGitToolSettings, ShellToolSettings
from ..tool_cycles import (
    UNLIMITED_TOOL_CYCLES,
    MaximumToolCycles,
)
from ..types import (
    assert_non_negative_int,
    assert_non_negative_number,
    assert_positive_int,
)
from ..utils import logger_replace

import gettext
import sys
from argparse import (
    SUPPRESS,
    ArgumentParser,
    ArgumentTypeError,
    Namespace,
    _ArgumentGroup,
    _SubParsersAction,
)
from asyncio import (
    FIRST_COMPLETED,
    all_tasks,
    ensure_future,
    gather,
    new_event_loop,
    set_event_loop,
    wait,
    wait_for,
)
from asyncio.exceptions import CancelledError
from collections.abc import Awaitable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import fields
from enum import StrEnum
from gettext import translation
from importlib import import_module
from importlib.util import find_spec
from locale import getlocale
from logging import (
    DEBUG,
    INFO,
    WARNING,
    Filter,
    Logger,
    LogRecord,
    basicConfig,
    getLogger,
)
from os import cpu_count, environ, getenv
from os.path import join
from pathlib import Path
from re import fullmatch
from signal import SIGINT, default_int_handler
from signal import signal as set_signal_handler
from subprocess import run
from threading import current_thread, main_thread
from tomllib import TOMLDecodeError
from tomllib import loads as toml_loads
from types import FrameType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    TextIO,
    TypeAlias,
    cast,
    get_args,
    get_origin,
)
from typing import get_args as get_type_args
from uuid import uuid4
from warnings import filterwarnings

from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Confirm, Prompt
from rich.theme import Theme as RichTheme

if TYPE_CHECKING:
    from ..cli.theme import Theme
else:
    Theme = Any

HubClient: TypeAlias = Any


class VectorFunction(StrEnum):
    COSINE_DISTANCE = "cosine_distance"
    INNER_PRODUCT = "inner_product"
    L1_DISTANCE = "l1_distance"
    L2_DISTANCE = "l2_distance"
    VECTOR_DIMS = "vector_dims"
    VECTOR_NORMS = "vector_norms"


_DEFAULT_SENTENCE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_SENTENCE_MODEL_MAX_TOKENS = 500
_DEFAULT_SENTENCE_MODEL_OVERLAP_SIZE = 125
_DEFAULT_SENTENCE_MODEL_WINDOW_SIZE = 250


class TransformerModel:
    """Provide the CLI's lightweight default-device hook."""

    @staticmethod
    def get_default_device() -> str:
        if _is_cuda_available():
            return "cuda"
        if _is_mps_available():
            return "mps"
        return "cpu"


class _AnonymousHub:
    domain = "huggingface.co"

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir

    def login(self) -> None:
        raise NotImplementedError()

    def user(self) -> User:
        raise NotImplementedError()


class Translator(Protocol):
    """Represent the translation helpers needed by the CLI."""

    def gettext(self, message: str) -> str: ...

    def ngettext(self, singular: str, plural: str, n: int) -> str: ...


_INTERRUPT_DRAIN_TIMEOUT = 0.5
_LOOP_HANDLES_SIGINT = False


def _task_run_json_stdout(args: Namespace) -> bool:
    task_json = (
        getattr(args, "command", None) == "task"
        and getattr(args, "task_command", None) == "run"
        and bool(getattr(args, "task_run_json", False))
    )
    flow_json = getattr(args, "command", None) == "flow" and (
        (
            getattr(args, "flow_command", None) == "run"
            and bool(getattr(args, "task_run_json", False))
        )
        or (
            getattr(args, "flow_command", None)
            in {
                "cancel",
                "compile",
                "graph",
                "inspect",
                "mermaid",
                "resume",
                "trace",
                "validate",
            }
            and bool(getattr(args, "flow_json", False))
        )
    )
    return task_json or flow_json


def _default_hf_cache_dir() -> str:
    return getenv("HF_HUB_CACHE") or "~/.cache/huggingface/hub"


def _module_exists(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except ValueError:
        return module_name in sys.modules


def _is_cuda_available() -> bool:
    if not _module_exists("torch"):
        return False
    try:
        cuda_module = import_module("torch.cuda")
    except ModuleNotFoundError:
        return False
    return bool(getattr(cuda_module, "is_available")())


def _is_mps_available() -> bool:
    if not _module_exists("torch"):
        return False
    try:
        mps_module = import_module("torch.backends.mps")
    except ModuleNotFoundError:
        return False
    return bool(getattr(mps_module, "is_available")())


def _cuda_device_count() -> int:
    if not _is_cuda_available():
        return 1
    cuda_module = import_module("torch.cuda")
    return int(getattr(cuda_module, "device_count")())


def is_available() -> bool:
    return _is_cuda_available()


def device_count() -> int:
    return _cuda_device_count()


def _set_cuda_device(index: int) -> None:
    if not _is_cuda_available():
        return
    cuda_module = import_module("torch.cuda")
    getattr(cuda_module, "set_device")(index)


def _destroy_torch_process_group() -> None:
    if not _module_exists("torch.distributed"):
        return
    dist_module = import_module("torch.distributed")
    getattr(dist_module, "destroy_process_group")()


def set_device(index: int) -> None:
    _set_cuda_device(index)


def destroy_process_group() -> None:
    _destroy_torch_process_group()


class _HFLoggingProxy:
    def get_logger(self, name: str) -> Logger:
        if not find_spec("transformers"):
            return getLogger(name)
        transformers_logging = import_module("transformers.utils.logging")
        return cast(Logger, getattr(transformers_logging, "get_logger")(name))


hf_logging = _HFLoggingProxy()


class _TaskArgumentParser(ArgumentParser):
    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: Namespace | None = None,
    ) -> Namespace:
        parsed, extras = self.parse_known_args(args, namespace)
        if extras and _consume_task_input_field_args(parsed, extras):
            return parsed
        if extras:
            self.error("unrecognized arguments")
        return parsed


def _consume_task_input_field_args(
    namespace: Namespace,
    extras: list[str],
) -> bool:
    if getattr(namespace, "command", None) not in {"flow", "task"}:
        return False
    if getattr(namespace, "command", None) == "task":
        if getattr(namespace, "task_command", None) not in {
            "enqueue",
            "run",
            "validate",
        }:
            return False
    elif getattr(namespace, "flow_command", None) != "run":
        return False
    fields: list[str] = []
    index = 0
    while index < len(extras):
        token = extras[index]
        if not token.startswith("--input-") or token == "--input-json":
            return False
        field_and_value = token[len("--input-") :]
        if "=" in field_and_value:
            field, value = field_and_value.split("=", 1)
        else:
            if index + 1 >= len(extras):
                return False
            value = extras[index + 1]
            if value.startswith("--"):
                return False
            field = field_and_value
            index += 1
        if not _valid_task_input_field(field):
            return False
        fields.append(f"{field}={value}")
        index += 1
    setattr(namespace, "task_input_fields", tuple(fields))
    return True


def _valid_task_input_field(value: str) -> bool:
    return bool(
        fullmatch(
            r"[A-Za-z][A-Za-z0-9_-]{0,63}" r"(\.[A-Za-z][A-Za-z0-9_-]{0,63})*",
            value,
        )
    )


def _transformers_utils_module() -> object | None:
    if not _module_exists("transformers"):
        return None
    return import_module("transformers.utils")


def is_flash_attn_2_available() -> bool:
    transformers_utils = _transformers_utils_module()
    if transformers_utils is None:
        return False
    return bool(
        cast(
            Callable[[], bool],
            getattr(
                transformers_utils,
                "is_flash_attn_2_available",
                lambda: False,
            ),
        )()
    )


def is_torch_flex_attn_available() -> bool:
    transformers_utils = _transformers_utils_module()
    if transformers_utils is None:
        return False
    return bool(
        cast(
            Callable[[], bool],
            getattr(
                transformers_utils,
                "is_torch_flex_attn_available",
                lambda: False,
            ),
        )()
    )


def _huggingface_hub_class() -> type[Any]:
    module = import_module("avalan.model.hubs.huggingface")
    return cast(type[Any], getattr(module, "HuggingfaceHub"))


def _load_command(module_name: str, function_name: str) -> Callable[..., Any]:
    module = import_module(module_name)
    return cast(Callable[..., Any], getattr(module, function_name))


async def agent_message_search(*args: Any, **kwargs: Any) -> Any:
    command = cast(
        Callable[..., Awaitable[Any]],
        _load_command("avalan.cli.commands.agent", "agent_message_search"),
    )
    return await command(*args, **kwargs)


async def agent_run(*args: Any, **kwargs: Any) -> Any:
    command = cast(
        Callable[..., Awaitable[Any]],
        _load_command("avalan.cli.commands.agent", "agent_run"),
    )
    return await command(*args, **kwargs)


async def agent_serve(*args: Any, **kwargs: Any) -> Any:
    command = cast(
        Callable[..., Awaitable[Any]],
        _load_command("avalan.cli.commands.agent", "agent_serve"),
    )
    return await command(*args, **kwargs)


async def agent_proxy(*args: Any, **kwargs: Any) -> Any:
    command = cast(
        Callable[..., Awaitable[Any]],
        _load_command("avalan.cli.commands.agent", "agent_proxy"),
    )
    return await command(*args, **kwargs)


async def agent_init(*args: Any, **kwargs: Any) -> Any:
    command = cast(
        Callable[..., Awaitable[Any]],
        _load_command("avalan.cli.commands.agent", "agent_init"),
    )
    return await command(*args, **kwargs)


def cache_delete(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.cache", "cache_delete")(
        *args, **kwargs
    )


def cache_download(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.cache", "cache_download")(
        *args, **kwargs
    )


def cache_list(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.cache", "cache_list")(
        *args, **kwargs
    )


async def memory_document_index(*args: Any, **kwargs: Any) -> Any:
    command = cast(
        Callable[..., Awaitable[Any]],
        _load_command("avalan.cli.commands.memory", "memory_document_index"),
    )
    return await command(*args, **kwargs)


async def memory_search(*args: Any, **kwargs: Any) -> Any:
    command = cast(
        Callable[..., Awaitable[Any]],
        _load_command("avalan.cli.commands.memory", "memory_search"),
    )
    return await command(*args, **kwargs)


async def memory_embeddings(*args: Any, **kwargs: Any) -> Any:
    command = cast(
        Callable[..., Awaitable[Any]],
        _load_command("avalan.cli.commands.memory", "memory_embeddings"),
    )
    return await command(*args, **kwargs)


def model_display(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.model", "model_display")(
        *args, **kwargs
    )


def model_install(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.model", "model_install")(
        *args, **kwargs
    )


async def model_run(*args: Any, **kwargs: Any) -> Any:
    command = cast(
        Callable[..., Awaitable[Any]],
        _load_command("avalan.cli.commands.model", "model_run"),
    )
    return await command(*args, **kwargs)


async def model_search(*args: Any, **kwargs: Any) -> Any:
    command = cast(
        Callable[..., Awaitable[Any]],
        _load_command("avalan.cli.commands.model", "model_search"),
    )
    return await command(*args, **kwargs)


def model_uninstall(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.model", "model_uninstall")(
        *args, **kwargs
    )


async def deploy_run(*args: Any, **kwargs: Any) -> Any:
    command = cast(
        Callable[..., Awaitable[Any]],
        _load_command("avalan.cli.commands.deploy", "deploy_run"),
    )
    return await command(*args, **kwargs)


def flow_run(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.flow", "flow_run")(
        *args, **kwargs
    )


def flow_inspect(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.flow", "flow_inspect")(
        *args, **kwargs
    )


def flow_trace(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.flow", "flow_trace")(
        *args, **kwargs
    )


def flow_cancel(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.flow", "flow_cancel")(
        *args, **kwargs
    )


def flow_compile(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.flow", "flow_compile")(
        *args, **kwargs
    )


def flow_graph(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.flow", "flow_graph")(
        *args, **kwargs
    )


def flow_resume(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.flow", "flow_resume")(
        *args, **kwargs
    )


def flow_validate(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.flow", "flow_validate")(
        *args, **kwargs
    )


def flow_mermaid(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.flow", "flow_mermaid")(
        *args, **kwargs
    )


def task_validate(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_validate")(
        *args, **kwargs
    )


def task_artifacts(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_artifacts")(
        *args, **kwargs
    )


def task_enqueue(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_enqueue")(
        *args, **kwargs
    )


def task_events(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_events")(
        *args, **kwargs
    )


def task_inspect(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_inspect")(
        *args, **kwargs
    )


def task_usage(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_usage")(
        *args, **kwargs
    )


def task_output(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_output")(
        *args, **kwargs
    )


def task_pgsql_check(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_pgsql_check")(
        *args, **kwargs
    )


def task_pgsql_diagnose(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_pgsql_diagnose")(
        *args, **kwargs
    )


def task_pgsql_migrate(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_pgsql_migrate")(
        *args, **kwargs
    )


def task_pgsql_stamp(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_pgsql_stamp")(
        *args, **kwargs
    )


def task_pgsql_status(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_pgsql_status")(
        *args, **kwargs
    )


def task_retention_sweep(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_retention_sweep")(
        *args, **kwargs
    )


def task_run(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_run")(
        *args, **kwargs
    )


def task_worker(*args: Any, **kwargs: Any) -> Any:
    return _load_command("avalan.cli.commands.task", "task_worker")(
        *args, **kwargs
    )


async def tokenize(*args: Any, **kwargs: Any) -> Any:
    command = cast(
        Callable[..., Awaitable[Any]],
        _load_command("avalan.cli.commands.tokenizer", "tokenize"),
    )
    return await command(*args, **kwargs)


@contextmanager
def _direct_keyboard_interrupts() -> Iterator[None]:
    if _LOOP_HANDLES_SIGINT or current_thread() is not main_thread():
        yield
        return

    previous_handler = set_signal_handler(SIGINT, default_int_handler)
    try:
        yield
    finally:
        set_signal_handler(SIGINT, previous_handler)


def run_in_loop(awaitable: Awaitable[object]) -> None:
    loop = new_event_loop()
    interrupted = False
    installed_signal_handler = False
    previous_signal_handler = None
    previous_loop_handles_sigint = _LOOP_HANDLES_SIGINT
    main_task = ensure_future(awaitable, loop=loop)
    interrupt_signal = loop.create_future()

    async def run_until_interrupt() -> object:
        done, _ = await wait(
            {main_task, interrupt_signal}, return_when=FIRST_COMPLETED
        )
        if interrupt_signal in done:
            if not main_task.done():
                main_task.cancel()
            raise KeyboardInterrupt()
        return await main_task

    def mark_interrupted() -> None:
        nonlocal interrupted
        interrupted = True

    def request_interrupt() -> None:
        mark_interrupted()
        if not interrupt_signal.done():
            interrupt_signal.set_result(None)
        if not main_task.done():
            main_task.cancel()

    def request_interrupt_from_signal(
        _signum: int, _frame: FrameType | None
    ) -> None:
        mark_interrupted()
        try:
            loop.call_soon_threadsafe(request_interrupt)
        except RuntimeError:
            request_interrupt()
        raise KeyboardInterrupt()

    try:
        set_event_loop(loop)
        if current_thread() is main_thread():
            previous_signal_handler = set_signal_handler(
                SIGINT, request_interrupt_from_signal
            )
            installed_signal_handler = True
        globals()["_LOOP_HANDLES_SIGINT"] = installed_signal_handler
        runner_task = ensure_future(run_until_interrupt(), loop=loop)
        try:
            loop.run_until_complete(runner_task)
        except (
            CancelledError,
            KeyboardInterrupt,
            CommandAbortException,
        ):
            interrupted = True
            raise
        finally:
            pending = [task for task in all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                pending_gather = gather(*pending, return_exceptions=True)
                if interrupted:
                    try:
                        loop.run_until_complete(
                            wait_for(
                                pending_gather,
                                timeout=_INTERRUPT_DRAIN_TIMEOUT,
                            )
                        )
                    except (
                        CancelledError,
                        KeyboardInterrupt,
                        TimeoutError,
                    ):
                        pass
                else:
                    loop.run_until_complete(pending_gather)
            if interrupted:
                try:
                    loop.run_until_complete(
                        wait_for(
                            loop.shutdown_asyncgens(),
                            timeout=_INTERRUPT_DRAIN_TIMEOUT,
                        )
                    )
                except (
                    CancelledError,
                    KeyboardInterrupt,
                    TimeoutError,
                ):
                    pass
            else:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            if main_task.done() and not main_task.cancelled():
                main_task.exception()
    finally:
        if installed_signal_handler:
            assert previous_signal_handler is not None
            set_signal_handler(SIGINT, previous_signal_handler)
        globals()["_LOOP_HANDLES_SIGINT"] = previous_loop_handles_sigint
        set_event_loop(None)
        loop.close()


class CLI:
    """Command line interface entry point."""

    _REFRESH_RATE = 4

    def __init__(self, logger: Logger):
        self._name = name()
        self._site = site()
        self._version = version()
        self._license = license()
        self._logger = logger
        self._abort_console: Console | None = None
        self._abort_printed = False
        self._abort_quiet = False
        self._abort_theme: Theme | None = None

        cache_dir = _default_hf_cache_dir()
        default_locale, _ = getlocale()
        default_locale = default_locale or "en_US"
        default_locales_path = join(
            Path(__file__).resolve().parents[3], "locale"
        )
        default_device = TransformerModel.get_default_device()
        self._parser = CLI._create_parser(
            default_device, cache_dir, default_locales_path, default_locale
        )

    @staticmethod
    def _default_parallel_count() -> int:
        return _cuda_device_count()

    @staticmethod
    def _default_flow_parallel_count() -> int:
        return cpu_count() or 1

    @staticmethod
    def _positive_integer(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ArgumentTypeError("must be an integer") from exc
        try:
            assert_positive_int(parsed, "value")
        except AssertionError as exc:
            raise ArgumentTypeError(str(exc)) from exc
        return parsed

    @staticmethod
    def _non_negative_integer(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ArgumentTypeError("must be an integer") from exc
        try:
            assert_non_negative_int(parsed, "value")
        except AssertionError as exc:
            raise ArgumentTypeError(str(exc)) from exc
        return parsed

    @staticmethod
    def _non_negative_number(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ArgumentTypeError("must be numeric") from exc
        try:
            assert_non_negative_number(parsed, "value")
        except AssertionError as exc:
            raise ArgumentTypeError(str(exc)) from exc
        return parsed

    @staticmethod
    def _maximum_tool_cycles(value: str) -> MaximumToolCycles:
        if value == UNLIMITED_TOOL_CYCLES:
            return value
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ArgumentTypeError(
                f"must be a positive integer or '{UNLIMITED_TOOL_CYCLES}'"
            ) from exc
        try:
            assert_positive_int(parsed, "value")
        except AssertionError as exc:
            raise ArgumentTypeError(
                f"must be a positive integer or '{UNLIMITED_TOOL_CYCLES}'"
            ) from exc
        return parsed

    @staticmethod
    def _default_attention(device: str) -> AttentionImplementation | None:
        try:
            if device.startswith("cuda") and is_available():
                if is_flash_attn_2_available():
                    return "flash_attention_2"
                if is_torch_flex_attn_available():
                    return "flex_attention"
            from torch.backends import mps

            if device.startswith("mps") and mps.is_available():
                return "sdpa"
        except Exception:
            pass
        return None

    @staticmethod
    def _create_parser(
        default_device: str,
        cache_dir: str,
        default_locales_path: str,
        default_locale: str,
    ) -> ArgumentParser:
        default_attention = CLI._default_attention(default_device)
        global_parser = ArgumentParser(add_help=False)
        global_parser.add_argument(
            "--cache-dir",
            default=cache_dir,
            type=str,
            help=(
                f"Path to huggingface cache hub (defaults to {cache_dir}, "
                "can also be specified with $HF_HUB_CACHE)"
            ),
        )
        global_parser.add_argument(
            "--subfolder",
            type=str,
            help="Subfolder inside model repository to load the model from",
        )
        global_parser.add_argument(
            "--tokenizer-subfolder",
            type=str,
            help=(
                "Subfolder inside model repository to load the tokenizer from"
            ),
        )
        global_parser.add_argument(
            "--device",
            type=str,
            required=False,
            default=default_device,
            help="Device to use (cpu, cuda, mps). Defaults to "
            + default_device,
        )
        global_parser.add_argument(
            "--parallel",
            type=str,
            choices=[p.value for p in ParallelStrategy],
            help="Tensor parallelism strategy to use",
        )
        global_parser.add_argument(
            "--parallel-count",
            type=int,
            default=CLI._default_parallel_count(),
            help=(
                "Number of processes to launch when --parallel is used "
                "(defaults to the number of available GPUs)"
            ),
        )
        global_parser.add_argument(
            "--disable-loading-progress-bar",
            action="store_true",
            default=False,
            help=(
                "If specified, the shard loading progress bar "
                "will not be shown"
            ),
        )
        global_parser.add_argument(
            "--hf-token",
            type=str,
            default=getenv("HF_TOKEN"),
            help="Your Huggingface access token",
        )
        global_parser.add_argument(
            "--locale",
            type=str,
            default=default_locale,
            help=f"Language to use (defaults to {default_locale})",
        )
        global_parser.add_argument(
            "--theme",
            choices=SUPPORTED_THEME_NAMES,
            default=SUPPRESS,
            help="Theme to use (default is fancy)",
        )
        global_parser.add_argument(
            "--loader-class",
            type=str,
            default="auto",
            choices=get_args(TextGenerationLoaderClass),
            help='Loader class to use (defaults to "auto")',
        )
        global_parser.add_argument(
            "--backend",
            type=str,
            default=Backend.TRANSFORMERS.value,
            choices=[b.value for b in Backend],
            help='Backend to use (defaults to "transformers")',
        )
        global_parser.add_argument(
            "--locales",
            type=str,
            default=default_locales_path,
            help=f"Path to locale files (defaults to {default_locales_path})",
        )
        global_parser.add_argument(
            "--low-cpu-mem-usage",
            action="store_true",
            default=False,
            help=(
                "If specified, loads the model using ~1x model size CPU memory"
            ),
        )
        global_parser.add_argument(
            "--login",
            action="store_true",
            help="Login to main hub (huggingface)",
        )
        global_parser.add_argument(
            "--no-repl",
            action="store_true",
            help="Don't echo input coming from stdin",
        )
        global_parser.add_argument(
            "--quiet",
            "-q",
            default=False,
            action="store_true",
            help=(
                "If specified, no welcome screen and only model output is "
                "displayed in model run (sets "
            )
            + ", ".join(
                [
                    "--disable-loading-progress-bar",
                    "--skip-hub-access-check",
                    "--skip-special-tokens",
                ]
            )
            + " automatically)",
        )
        global_parser.add_argument(
            "--tty",
            default="/dev/tty",
            help="TTY stream to use for interactive prompts",
        )
        global_parser.add_argument(
            "--record",
            action="store_true",
            default=False,
            help=(
                "If specified, the current console output will be regularly "
                "saved to SVG files."
            ),
        )
        global_parser.add_argument(
            "--revision",
            type=str,
            help="Model revision to use",
        )
        global_parser.add_argument(
            "--skip-hub-access-check",
            action="store_true",
            default=False,
            help="If specified, skip hub model access check",
        )
        global_parser.add_argument(
            "--verbose", "-v", action="count", help="Set verbosity"
        )
        global_parser.add_argument(
            "--version",
            action="store_true",
            help="Display this program's version, and exit",
        )

        global_parser.add_argument(
            "--weight-type",
            type=str,
            choices=get_args(WeightType),
            help="Weight type to use (defaults to best available)",
        )

        parser: ArgumentParser = _TaskArgumentParser(
            description="Avalan CLI", parents=[global_parser]
        )

        command_parsers = parser.add_subparsers(dest="command")

        # Memory options shared by commands: memory embeddings, memory document
        memory_partitions_parser = ArgumentParser(add_help=False)
        memory_partitions_display_group = (
            memory_partitions_parser.add_mutually_exclusive_group()
        )
        memory_partitions_display_group.add_argument(
            "--no-display-partitions",
            action="store_true",
            default=False,
            help="If specified, don't display memory partitions",
        )
        memory_partitions_display_group.add_argument(
            "--display-partitions",
            default=6,
            type=int,
            help="Display up to this many partitions, if more summarize",
        )
        memory_partitions_parser.add_argument(
            "--partition",
            action="store_true",
            default=False,
            help="If specified, partition string",
        )
        memory_partitions_parser.add_argument(
            "--partition-max-tokens",
            default=500,
            type=int,
            help="Maximum number of tokens to allow on each partition",
        )
        memory_partitions_parser.add_argument(
            "--partition-overlap",
            default=125,
            type=int,
            help=(
                "How many tokens can potentially overlap in "
                "different partitions"
            ),
        )
        memory_partitions_parser.add_argument(
            "--partition-window",
            default=250,
            type=int,
            help="Number of tokens per window when partitioning",
        )

        # Model options shared by commands: cache download, model install
        model_install_parser = ArgumentParser(add_help=False)
        model_install_parser.add_argument(
            "model",
            type=str,
            help="Model to download",
        )
        model_install_parser.add_argument(
            "--workers",
            default=8,
            type=int,
            help="How many download workers to use",
        )
        model_install_parser.add_argument(
            "--local-dir",
            type=str,
            help="Local directory to download the model to",
        )
        model_install_parser.add_argument(
            "--local-dir-symlinks",
            action="store_true",
            default=None,
            help="Use symlinks when downloading to local dir",
        )

        # Model options shared by commands: memory embeddings, model
        model_options_parser = ArgumentParser(add_help=False)
        model_options_parser.add_argument(
            "model",
            type=str,
            help="Model to use",
        )
        model_options_parser.add_argument(
            "--base-url",
            type=str,
            help=(
                "If specified and model is a vendor model that supports it,"
                "load model using the given base URL"
            ),
        )
        model_options_parser.add_argument(
            "--load",
            action="store_true",
            help="If specified, load model and show more information",
        )
        model_options_parser.add_argument(
            "--special-token",
            type=str,
            action="append",
            help=(
                "Special token to add to tokenizer, only when model is loaded"
            ),
        )
        model_options_parser.add_argument(
            "--token",
            type=str,
            action="append",
            help="Token to add to tokenizer, only when model is loaded",
        )
        model_options_parser.add_argument(
            "--tokenizer",
            type=str,
            help=(
                "Path to tokenizer to use instead of model's default, only "
                "if model is loaded"
            ),
        )

        # Inference options shared by commands: agent run, model run
        model_inference_display_parser = ArgumentParser(add_help=False)
        model_inference_display_parser.add_argument(
            "--display-events",
            action="store_true",
            help=(
                "Show non-tool stream events when an orchestrator or agent is "
                "involved."
            ),
        )
        model_inference_display_parser.add_argument(
            "--stats",
            action="store_true",
            default=False,
            help="Show token generation statistics for streaming output",
        )
        model_inference_display_parser.add_argument(
            "--display-pause",
            type=int,
            nargs="?",
            const=500,  # 500 is the default if argument present but no value
            default=None,
            help=(
                "Pause (in ms.) when cycling through selected tokens as "
                "defined by --display-probabilities"
            ),
        )
        model_inference_display_parser.add_argument(
            "--display-probabilities",
            action="store_true",
            help=(
                "If --display-tokens specified, show also the token "
                "probability distribution"
            ),
        )
        model_inference_display_parser.add_argument(
            "--display-probabilities-maximum",
            type=float,
            default=0.8,
            help=(
                "When --display-probabilities is used, select tokens which "
                "logit probability is no higher than this value. "
                "Defaults to 0.8"
            ),
        )
        model_inference_display_parser.add_argument(
            "--display-probabilities-sample-minimum",
            type=float,
            default=0.1,
            help=(
                "When --display-probabilities is used, select tokens that "
                "have alternate tokens with a logit probability at least or "
                "higher than this value. Defaults to 0.1"
            ),
        )
        model_inference_display_parser.add_argument(
            "--display-time-to-n-token",
            type=int,
            nargs="?",
            const=256,  # 256 is the default if argument present but no value
            default=None,
            help=(
                "Display the time it takes to reach the given Nth token "
                "(defaults to 256)"
            ),
        )
        model_inference_display_parser.add_argument(
            "--skip-display-reasoning-time",
            action="store_true",
            help="Don't display total reasoning time",
        )
        model_inference_display_parser.add_argument(
            "--display-reasoning",
            action="store_true",
            help="Display streamed reasoning text in the live response panel",
        )
        model_inference_display_parser.add_argument(
            "--display-tokens",
            type=int,
            nargs="?",
            const=15,  # 15 is the default if argument present but no value
            default=None,
            help="How many tokens with full information to display at a time",
        )
        model_inference_display_parser.add_argument(
            "--display-tools",
            action="store_true",
            help="Show tool lifecycle details for agent or orchestrator runs.",
        )
        model_inference_display_parser.add_argument(
            "--display-tools-events",
            type=int,
            default=None,
            help=(
                "How many tool events to show on tool call panel. "
                "Defaults to all retained tool events; use 0 to hide "
                "completed tool history."
            ),
        )

        display_answer_height_group = (
            model_inference_display_parser.add_mutually_exclusive_group()
        )
        display_answer_height_group.add_argument(
            "--display-answer-height-expand",
            action="store_true",
            help="Expand answer section to full height",
        )
        display_answer_height_group.add_argument(
            "--display-answer-height",
            type=int,
            default=12,
            help="Height of the answer section (defaults to 12)",
        )

        # Agent command
        agent_parser = command_parsers.add_parser(
            name="agent",
            description="Manage AI agents",
            parents=[global_parser],
        )
        agent_command_parsers = agent_parser.add_subparsers(
            dest="agent_command"
        )

        agent_message_parser = agent_command_parsers.add_parser(
            name="message",
            description="Manage AI agent messages",
            parents=[global_parser],
        )
        agent_message_command_parsers = agent_message_parser.add_subparsers(
            dest="agent_message_command"
        )

        agent_message_search_parser = agent_message_command_parsers.add_parser(
            name="search",
            description="Search within an agent's message memory",
            parents=[global_parser],
        )
        agent_message_search_parser.add_argument(
            "specifications_file",
            type=str,
            nargs="?",
            help="File that holds the agent specifications",
        )
        agent_message_search_parser.add_argument(
            "--function",
            type=VectorFunction,
            choices=list(VectorFunction),
            required=True,
            default=VectorFunction.L2_DISTANCE,
            help="Vector function to use for searching",
        )
        agent_message_search_parser.add_argument(
            "--id", type=str, required=True
        )
        agent_message_search_parser.add_argument(
            "--limit",
            type=int,
            help="If specified, load up to these many recent messages",
        )
        agent_message_search_parser.add_argument(
            "--participant",
            type=str,
            required=True,
            help="Search messages with given participant",
        )
        agent_message_search_parser.add_argument(
            "--session",
            type=str,
            required=True,
            help="Search within the given session",
        )
        CLI._add_agent_settings_arguments(agent_message_search_parser)
        CLI._add_tool_settings_arguments(
            agent_message_search_parser,
            prefix="browser",
            settings_cls=BrowserToolSettings,
        )
        CLI._add_tool_settings_arguments(
            agent_message_search_parser,
            prefix="database",
            settings_cls=DatabaseToolSettings,
        )
        CLI._add_tool_settings_arguments(
            agent_message_search_parser,
            prefix="shell",
            settings_cls=ShellToolSettings,
        )
        CLI._add_skills_settings_arguments(agent_message_search_parser)

        agent_common_parser = ArgumentParser(add_help=False)
        agent_common_parser.add_argument(
            "specifications_file",
            type=str,
            nargs="?",
            help="File that holds the agent specifications",
        )
        agent_common_parser.add_argument(
            "--id", type=str, help="Use given ID as the agent ID"
        )
        agent_common_parser.add_argument(
            "--participant",
            default=uuid4(),
            help=(
                "If specified, this is the participant ID interacting with "
                "the agent"
            ),
        )

        agent_run_parser = agent_command_parsers.add_parser(
            name="run",
            description="Run an AI agent",
            parents=[
                global_parser,
                model_inference_display_parser,
                agent_common_parser,
            ],
        )
        agent_run_parser.add_argument(
            "--conversation",
            action="store_true",
            default=False,
            help="Activate conversation mode with the agent",
        )
        agent_run_parser.add_argument(
            "--watch",
            action="store_true",
            default=False,
            help=(
                "Reload agent when the specification file changes "
                "(only with --conversation)"
            ),
        )
        agent_session_group = agent_run_parser.add_mutually_exclusive_group()
        agent_session_group.add_argument(
            "--no-session",
            action="store_true",
            default=False,
            help=(
                "If specified, don't use sessions in persistent message memory"
            ),
        )
        agent_session_group.add_argument(
            "--session",
            type=str,
            help="Continue the conversation on the given session",
        )

        agent_run_parser.add_argument(
            "--skip-load-recent-messages",
            default=False,
            action="store_true",
            help="If specified, skips loading recent messages",
        )
        agent_run_parser.add_argument(
            "--load-recent-messages-limit",
            type=int,
            help="If specified, load up to these many recent messages",
        )
        agent_run_parser.add_argument(
            "--sync",
            dest="use_sync_generator",
            action="store_true",
            default=False,
            help="Don't use an async generator (streaming output)",
        )
        agent_run_parser.add_argument(
            "--tools-confirm",
            action="store_true",
            help="Confirm tool calls before execution",
        )
        agent_run_parser.add_argument(
            "--input-file",
            action="append",
            help=(
                "Attach a local file as native input for text generation. "
                "May be specified multiple times."
            ),
        )
        agent_run_parser.add_argument(
            "--tool-format",
            type=str,
            choices=[t.value for t in ToolFormat],
            help="Tool format",
        )
        agent_run_parser.add_argument(
            "--tool-choice",
            type=str,
            help="Force a tool by canonical name when supported.",
        )
        agent_run_parser.add_argument(
            "--tool-recovery-format",
            action="append",
            choices=[t.value for t in ToolCallRecoveryFormat],
            help="Enable a tool-call recovery format",
        )
        CLI._add_tool_name_policy_arguments(agent_run_parser)
        agent_run_parser.add_argument(
            "--reasoning-tag",
            type=str,
            choices=[t.value for t in ReasoningTag],
            help="Reasoning tag style",
        )
        CLI._add_ds4_backend_options(agent_run_parser)

        CLI._add_agent_settings_arguments(agent_run_parser)
        CLI._add_tool_settings_arguments(
            agent_run_parser,
            prefix="browser",
            settings_cls=BrowserToolSettings,
        )
        CLI._add_tool_settings_arguments(
            agent_run_parser,
            prefix="database",
            settings_cls=DatabaseToolSettings,
        )
        CLI._add_tool_settings_arguments(
            agent_run_parser,
            prefix="graph",
            settings_cls=GraphToolSettings,
        )
        CLI._add_tool_settings_arguments(
            agent_run_parser,
            prefix="shell",
            settings_cls=ShellToolSettings,
        )
        CLI._add_skills_settings_arguments(agent_run_parser)

        agent_serve_parser = agent_command_parsers.add_parser(
            name="serve",
            description="Serve an AI agent as an API endpoint",
            parents=[global_parser, agent_common_parser],
        )
        CLI._add_agent_server_arguments(agent_serve_parser)
        CLI._add_agent_settings_arguments(agent_serve_parser)
        CLI._add_tool_name_policy_arguments(agent_serve_parser)
        CLI._add_tool_settings_arguments(
            agent_serve_parser,
            prefix="browser",
            settings_cls=BrowserToolSettings,
        )
        CLI._add_tool_settings_arguments(
            agent_serve_parser,
            prefix="database",
            settings_cls=DatabaseToolSettings,
        )
        CLI._add_tool_settings_arguments(
            agent_serve_parser,
            prefix="graph",
            settings_cls=GraphToolSettings,
        )
        CLI._add_tool_settings_arguments(
            agent_serve_parser,
            prefix="shell",
            settings_cls=ShellToolSettings,
        )
        CLI._add_skills_settings_arguments(agent_serve_parser)

        agent_proxy_parser = agent_command_parsers.add_parser(
            name="proxy",
            description="Serve a proxy agent as an API endpoint",
            parents=[global_parser, agent_common_parser],
        )
        CLI._add_agent_server_arguments(agent_proxy_parser)
        CLI._add_agent_settings_arguments(agent_proxy_parser)
        CLI._add_tool_name_policy_arguments(agent_proxy_parser)
        CLI._add_tool_settings_arguments(
            agent_proxy_parser,
            prefix="browser",
            settings_cls=BrowserToolSettings,
        )
        CLI._add_tool_settings_arguments(
            agent_proxy_parser,
            prefix="database",
            settings_cls=DatabaseToolSettings,
        )
        CLI._add_tool_settings_arguments(
            agent_proxy_parser,
            prefix="graph",
            settings_cls=GraphToolSettings,
        )
        CLI._add_tool_settings_arguments(
            agent_proxy_parser,
            prefix="shell",
            settings_cls=ShellToolSettings,
        )
        CLI._add_skills_settings_arguments(agent_proxy_parser)

        agent_init_parser = agent_command_parsers.add_parser(
            name="init",
            description="Create an agent definition",
            parents=[global_parser],
        )
        CLI._add_agent_settings_arguments(agent_init_parser)
        CLI._add_tool_name_policy_arguments(agent_init_parser)
        CLI._add_tool_settings_arguments(
            agent_init_parser,
            prefix="graph",
            settings_cls=GraphToolSettings,
        )
        CLI._add_tool_settings_arguments(
            agent_init_parser,
            prefix="shell",
            settings_cls=ShellToolSettings,
        )
        CLI._add_skills_settings_arguments(agent_init_parser)

        # Cache command
        cache_parser = command_parsers.add_parser(
            name="cache",
            description="Manage models cache",
            parents=[global_parser],
        )
        cache_command_parsers = cache_parser.add_subparsers(
            dest="cache_command"
        )
        cache_delete_parser = cache_command_parsers.add_parser(
            name="delete",
            description="Delete cached model data",
            parents=[global_parser],
        )
        cache_delete_parser.add_argument(
            "--delete",
            action="store_true",
            help=(
                "Actually delete. If not provided, a dry run is performed "
                "and data that would be deleted is shown, yet not deleted"
            ),
        )
        cache_delete_parser.add_argument(
            "--model",
            "-m",
            type=str,
            required=True,
            help="Model to delete",
        )
        cache_delete_parser.add_argument(
            "--delete-revision",
            type=str,
            action="append",
            help="Revision to delete",
        )
        cache_command_parsers.add_parser(
            name="download",
            description="Download model data to cache",
            parents=[global_parser, model_install_parser],
        )
        cache_list_parser = cache_command_parsers.add_parser(
            name="list",
            description="List cache contents",
            parents=[global_parser],
        )
        cache_list_parser.add_argument(
            "--model",
            type=str,
            action="append",
            help="Models to show content details on",
        )
        cache_list_parser.add_argument(
            "--summary",
            action="store_true",
            help=(
                "If specified, when showing one or more models show only "
                "summary"
            ),
        )

        # Deploy command
        deploy_parser = command_parsers.add_parser(
            name="deploy",
            description="Manage AI deployments",
            parents=[global_parser],
        )
        deploy_command_parsers = deploy_parser.add_subparsers(
            dest="deploy_command"
        )
        deploy_run_parser = deploy_command_parsers.add_parser(
            name="run",
            description="Perform a deployment",
            parents=[global_parser],
        )
        deploy_run_parser.add_argument(
            "deployment",
            type=str,
            help="Deployment to run",
        )

        # Flow command
        flow_parser = command_parsers.add_parser(
            name="flow", description="Manage AI flows", parents=[global_parser]
        )
        flow_command_parsers = flow_parser.add_subparsers(dest="flow_command")
        flow_run_parser = flow_command_parsers.add_parser(
            name="run",
            description="Run a given flow",
            parents=[global_parser],
        )
        flow_run_parser.add_argument(
            "flow",
            type=str,
            help="Flow to run",
        )
        flow_run_parser.add_argument(
            "--input",
            dest="task_input",
            type=str,
            default=None,
            help="Flow input value.",
        )
        flow_run_parser.add_argument(
            "--input-json",
            dest="task_input_json",
            type=str,
            default=None,
            help="Flow input JSON value or @file.",
        )
        flow_run_parser.add_argument(
            "--file",
            dest="task_files",
            action="append",
            default=None,
            help="Attach a local flow input file as field=path.",
        )
        flow_run_parser.add_argument(
            "--file-mime",
            dest="task_file_mime_types",
            action="append",
            default=None,
            help="Set a flow file MIME hint as field=mime/type.",
        )
        flow_run_parser.add_argument(
            "--pdf",
            dest="task_pdf",
            type=str,
            default=None,
            help="Attach one top-level PDF file input.",
        )
        flow_run_parser.add_argument(
            "--json",
            dest="task_run_json",
            action="store_true",
            help="Print successful flow output as compact JSON.",
        )
        flow_run_parser.add_argument(
            "--output",
            dest="task_output_path",
            type=str,
            default=None,
            help="Write successful flow output to a JSON file.",
        )
        flow_run_parser.add_argument(
            "--flow-parallel",
            dest="flow_parallel",
            metavar="N",
            type=CLI._positive_integer,
            default=CLI._default_flow_parallel_count(),
            help=(
                "Maximum number of ready flow nodes to execute in parallel "
                "(defaults to the number of CPUs)"
            ),
        )
        flow_run_parser.add_argument(
            "--tool",
            type=str,
            action="append",
            help="Enable a tool for strict flow tool nodes.",
        )
        flow_run_parser.add_argument(
            "--tools",
            type=str,
            action="append",
            help=(
                "Enable tools matching a namespace for strict flow tool nodes."
            ),
        )
        CLI._add_tool_name_policy_arguments(flow_run_parser)
        CLI._add_tool_settings_arguments(
            flow_run_parser,
            prefix="browser",
            settings_cls=BrowserToolSettings,
        )
        CLI._add_tool_settings_arguments(
            flow_run_parser,
            prefix="database",
            settings_cls=DatabaseToolSettings,
        )
        CLI._add_tool_settings_arguments(
            flow_run_parser,
            prefix="graph",
            settings_cls=GraphToolSettings,
        )
        CLI._add_tool_settings_arguments(
            flow_run_parser,
            prefix="shell",
            settings_cls=ShellToolSettings,
        )
        flow_inspect_parser = flow_command_parsers.add_parser(
            name="inspect",
            description="Inspect a durable flow run",
            parents=[global_parser],
        )
        flow_inspect_parser.add_argument(
            "run_id",
            type=str,
            help="Task run id to inspect",
        )
        flow_inspect_parser.add_argument(
            "--store-dsn",
            type=str,
            default=None,
            help="Durable task store PostgreSQL DSN.",
        )
        flow_inspect_parser.add_argument(
            "--store-schema",
            type=str,
            default=None,
            help="Durable task store PostgreSQL schema.",
        )
        flow_inspect_parser.add_argument(
            "--after-sequence",
            type=int,
            default=None,
            help="Only include task events after this sequence.",
        )
        flow_inspect_parser.add_argument(
            "--json",
            dest="flow_json",
            action="store_true",
            help="Print flow inspection as compact JSON.",
        )
        flow_trace_parser = flow_command_parsers.add_parser(
            name="trace",
            description="Export a sanitized flow trace",
            parents=[global_parser],
        )
        flow_trace_parser.add_argument(
            "run_id",
            type=str,
            help="Task run id to export",
        )
        flow_trace_parser.add_argument(
            "--store-dsn",
            type=str,
            default=None,
            help="Durable task store PostgreSQL DSN.",
        )
        flow_trace_parser.add_argument(
            "--store-schema",
            type=str,
            default=None,
            help="Durable task store PostgreSQL schema.",
        )
        flow_trace_parser.add_argument(
            "--after-sequence",
            type=int,
            default=None,
            help="Only include task events after this sequence.",
        )
        flow_trace_parser.add_argument(
            "--json",
            dest="flow_json",
            action="store_true",
            help="Print sanitized flow trace as compact JSON.",
        )
        flow_cancel_parser = flow_command_parsers.add_parser(
            name="cancel",
            description="Request cancellation for a durable flow run",
            parents=[global_parser],
        )
        flow_cancel_parser.add_argument(
            "run_id",
            type=str,
            help="Task run id to cancel",
        )
        flow_cancel_parser.add_argument(
            "--store-dsn",
            type=str,
            default=None,
            help="Durable task store PostgreSQL DSN.",
        )
        flow_cancel_parser.add_argument(
            "--store-schema",
            type=str,
            default=None,
            help="Durable task store PostgreSQL schema.",
        )
        flow_cancel_parser.add_argument(
            "--json",
            dest="flow_json",
            action="store_true",
            help="Print cancellation result as compact JSON.",
        )
        flow_resume_parser = flow_command_parsers.add_parser(
            name="resume",
            description="Resume a paused strict flow",
            parents=[global_parser],
        )
        flow_resume_parser.add_argument(
            "flow",
            type=str,
            help="Flow definition TOML file to resume",
        )
        flow_resume_parser.add_argument(
            "run_id",
            type=str,
            help="Task run id to resume",
        )
        flow_resume_parser.add_argument(
            "--decision-json",
            required=True,
            help="JSON object mapping review nodes to decision payloads.",
        )
        flow_resume_parser.add_argument(
            "--store-dsn",
            type=str,
            default=None,
            help="Durable task store PostgreSQL DSN.",
        )
        flow_resume_parser.add_argument(
            "--store-schema",
            type=str,
            default=None,
            help="Durable task store PostgreSQL schema.",
        )
        flow_resume_parser.add_argument(
            "--json",
            dest="flow_json",
            action="store_true",
            help="Print resumed flow output as compact JSON.",
        )
        flow_validate_parser = flow_command_parsers.add_parser(
            name="validate",
            description="Validate a flow definition",
            parents=[global_parser],
        )
        flow_validate_parser.add_argument(
            "flow",
            type=str,
            help="Flow definition TOML file to validate",
        )
        flow_validate_parser.add_argument(
            "--encoding",
            default="utf-8",
            help="File encoding used when reading local flow files.",
        )
        flow_validate_parser.add_argument(
            "--json",
            dest="flow_json",
            action="store_true",
            help="Print diagnostics as compact JSON.",
        )
        flow_compile_parser = flow_command_parsers.add_parser(
            name="compile",
            description="Compile a flow definition to strict TOML",
            parents=[global_parser],
        )
        flow_compile_parser.add_argument(
            "flow",
            type=str,
            help="Flow definition TOML file to compile",
        )
        flow_compile_parser.add_argument(
            "--encoding",
            default="utf-8",
            help="File encoding used when reading local flow files.",
        )
        flow_compile_output_group = (
            flow_compile_parser.add_mutually_exclusive_group()
        )
        flow_compile_output_group.add_argument(
            "--output",
            type=str,
            default=None,
            help="Write canonical strict TOML to this path.",
        )
        flow_compile_output_group.add_argument(
            "--check",
            action="store_true",
            help="Compile and validate without writing output.",
        )
        flow_compile_parser.add_argument(
            "--json",
            dest="flow_json",
            action="store_true",
            help="Print compile status as compact JSON.",
        )
        flow_graph_parser = flow_command_parsers.add_parser(
            name="graph",
            description="Inspect flow authoring graphs",
            parents=[global_parser],
        )
        flow_graph_command_parsers = flow_graph_parser.add_subparsers(
            dest="flow_graph_command",
            required=True,
        )
        flow_graph_inspect_parser = flow_graph_command_parsers.add_parser(
            name="inspect",
            description="Inspect a flow authoring graph",
            parents=[global_parser],
        )
        flow_graph_inspect_parser.add_argument(
            "flow",
            type=str,
            help="Flow definition TOML file to inspect",
        )
        flow_graph_inspect_parser.add_argument(
            "--encoding",
            default="utf-8",
            help="File encoding used when reading local flow files.",
        )
        flow_graph_inspect_parser.add_argument(
            "--json",
            dest="flow_json",
            action="store_true",
            help="Print graph inspection as compact JSON.",
        )
        flow_mermaid_parser = flow_command_parsers.add_parser(
            name="mermaid",
            description="Inspect and render Mermaid flow views",
            parents=[global_parser],
        )
        flow_mermaid_command_parsers = flow_mermaid_parser.add_subparsers(
            dest="flow_mermaid_command",
            required=True,
        )
        flow_mermaid_parse_parser = flow_mermaid_command_parsers.add_parser(
            name="parse",
            description="Parse a Mermaid flow view",
            parents=[global_parser],
        )
        flow_mermaid_parse_parser.add_argument(
            "diagram",
            type=str,
            help="Mermaid diagram file to parse",
        )
        flow_mermaid_parse_parser.add_argument(
            "--encoding",
            default="utf-8",
            help="File encoding used when reading local Mermaid files.",
        )
        flow_mermaid_parse_parser.add_argument(
            "--mode",
            choices=("presentation", "executable"),
            required=True,
            help="Mermaid import mode.",
        )
        flow_mermaid_parse_parser.add_argument(
            "--json",
            dest="flow_json",
            action="store_true",
            help="Print parsed view and diagnostics as compact JSON.",
        )
        flow_mermaid_render_parser = flow_mermaid_command_parsers.add_parser(
            name="render",
            description="Render a safe Mermaid flow view",
            parents=[global_parser],
        )
        flow_mermaid_render_parser.add_argument(
            "diagram",
            type=str,
            help="Mermaid diagram file to render",
        )
        flow_mermaid_render_parser.add_argument(
            "--encoding",
            default="utf-8",
            help="File encoding used when reading local Mermaid files.",
        )
        flow_mermaid_render_parser.add_argument(
            "--mode",
            choices=("presentation", "executable"),
            required=True,
            help="Mermaid import mode.",
        )
        flow_mermaid_render_parser.add_argument(
            "--json",
            dest="flow_json",
            action="store_true",
            help="Print rendered source and diagnostics as compact JSON.",
        )
        flow_mermaid_compare_parser = flow_mermaid_command_parsers.add_parser(
            name="compare",
            description="Compare a Mermaid flow view with a flow definition",
            parents=[global_parser],
        )
        flow_mermaid_compare_parser.add_argument(
            "diagram",
            type=str,
            help="Mermaid diagram file to compare",
        )
        flow_mermaid_compare_parser.add_argument(
            "flow",
            type=str,
            help="Flow definition TOML file to compare",
        )
        flow_mermaid_compare_parser.add_argument(
            "--encoding",
            default="utf-8",
            help="File encoding used when reading local flow files.",
        )
        flow_mermaid_compare_parser.add_argument(
            "--mode",
            choices=("presentation", "executable"),
            required=True,
            help="Mermaid import mode.",
        )
        flow_mermaid_compare_parser.add_argument(
            "--json",
            dest="flow_json",
            action="store_true",
            help="Print comparison diagnostics as compact JSON.",
        )
        flow_mermaid_skeleton_parser = flow_mermaid_command_parsers.add_parser(
            name="skeleton",
            description="Create a non-executing flow skeleton",
            parents=[global_parser],
        )
        flow_mermaid_skeleton_parser.add_argument(
            "diagram",
            type=str,
            help="Mermaid diagram file to skeletonize",
        )
        flow_mermaid_skeleton_parser.add_argument(
            "--encoding",
            default="utf-8",
            help="File encoding used when reading local Mermaid files.",
        )
        flow_mermaid_skeleton_parser.add_argument(
            "--mode",
            choices=("presentation", "executable"),
            required=True,
            help="Mermaid import mode.",
        )
        flow_mermaid_skeleton_parser.add_argument(
            "--name",
            required=True,
            help="Skeleton flow name.",
        )
        flow_mermaid_skeleton_parser.add_argument(
            "--flow-version",
            dest="version",
            default=None,
            help="Skeleton flow version.",
        )
        flow_mermaid_skeleton_parser.add_argument(
            "--flow-revision",
            dest="revision",
            default=None,
            help="Skeleton flow revision.",
        )
        flow_mermaid_skeleton_parser.add_argument(
            "--json",
            dest="flow_json",
            action="store_true",
            help="Print skeleton definition and diagnostics as compact JSON.",
        )

        # Task command
        task_parser = command_parsers.add_parser(
            name="task",
            description="Manage intelligence tasks",
            parents=[global_parser],
        )
        task_command_parsers = task_parser.add_subparsers(dest="task_command")
        task_input_parser = ArgumentParser(add_help=False)
        task_input_parser.add_argument(
            "--input",
            dest="task_input",
            type=str,
            default=None,
            help="Task input value.",
        )
        task_input_parser.add_argument(
            "--input-json",
            dest="task_input_json",
            type=str,
            default=None,
            help="Task input JSON value or @file.",
        )
        task_input_parser.add_argument(
            "--file",
            dest="task_files",
            action="append",
            default=None,
            help="Attach a local task input file as field=path.",
        )
        task_input_parser.add_argument(
            "--file-descriptor",
            dest="task_file_descriptors",
            action="append",
            default=None,
            help="Attach an explicit task file descriptor as field=json.",
        )
        task_input_parser.add_argument(
            "--provider-file-id",
            dest="task_provider_file_ids",
            action="append",
            default=None,
            help="Attach a provider file id as field=provider:reference.",
        )
        task_input_parser.add_argument(
            "--hosted-url",
            dest="task_hosted_urls",
            action="append",
            default=None,
            help="Attach a provider-hosted URL as field=provider:url.",
        )
        task_input_parser.add_argument(
            "--object-store-uri",
            dest="task_object_store_uris",
            action="append",
            default=None,
            help="Attach an object-store URI as field=provider:uri.",
        )
        task_input_parser.add_argument(
            "--file-mime",
            dest="task_file_mime_types",
            action="append",
            default=None,
            help="Set a task file MIME hint as field=mime/type.",
        )
        task_input_parser.add_argument(
            "--file-role",
            dest="task_file_roles",
            action="append",
            default=None,
            help="Set a task file role hint as field=role.",
        )
        task_input_parser.add_argument(
            "--file-size",
            dest="task_file_sizes",
            action="append",
            default=None,
            help="Set a task file size hint as field=bytes.",
        )
        task_input_parser.add_argument(
            "--file-sha256",
            dest="task_file_sha256",
            action="append",
            default=None,
            help="Set a task file SHA-256 digest hint as field=hex.",
        )
        task_input_parser.add_argument(
            "--file-conversion",
            dest="task_file_conversions",
            action="append",
            default=None,
            help="Request a descriptor conversion as field=name[:json].",
        )
        task_tool_parser = ArgumentParser(add_help=False)
        task_tool_parser.add_argument(
            "--tool",
            type=str,
            action="append",
            help="Enable a tool for task agent or strict flow tool nodes.",
        )
        task_tool_parser.add_argument(
            "--tools",
            type=str,
            action="append",
            help=(
                "Enable tools matching a namespace for task agent or strict "
                "flow tool nodes."
            ),
        )
        CLI._add_tool_name_policy_arguments(task_tool_parser)
        CLI._add_tool_settings_arguments(
            task_tool_parser,
            prefix="browser",
            settings_cls=BrowserToolSettings,
        )
        CLI._add_tool_settings_arguments(
            task_tool_parser,
            prefix="database",
            settings_cls=DatabaseToolSettings,
        )
        CLI._add_tool_settings_arguments(
            task_tool_parser,
            prefix="graph",
            settings_cls=GraphToolSettings,
        )
        CLI._add_tool_settings_arguments(
            task_tool_parser,
            prefix="shell",
            settings_cls=ShellToolSettings,
        )
        task_validate_parser = task_command_parsers.add_parser(
            name="validate",
            description="Validate an intelligence task definition",
            parents=[global_parser, task_input_parser, task_tool_parser],
        )
        task_validate_parser.add_argument(
            "definition",
            type=str,
            help="Task definition TOML file to validate",
        )
        task_run_parser = task_command_parsers.add_parser(
            name="run",
            description="Run an intelligence task directly",
            parents=[global_parser, task_input_parser, task_tool_parser],
        )
        task_run_parser.add_argument(
            "definition",
            type=str,
            help="Task definition TOML file to run",
        )
        task_run_parser.add_argument(
            "--store-dsn",
            type=str,
            default=None,
            help="Durable task store PostgreSQL DSN.",
        )
        task_run_parser.add_argument(
            "--store-schema",
            type=str,
            default=None,
            help="Durable task store PostgreSQL schema.",
        )
        task_run_parser.add_argument(
            "--ephemeral",
            action="store_true",
            help="Use a non-durable in-memory store for this direct run.",
        )
        task_run_parser.add_argument(
            "--json",
            dest="task_run_json",
            action="store_true",
            help="Print successful structured task output as compact JSON.",
        )
        task_run_parser.add_argument(
            "--output",
            dest="task_output_path",
            type=str,
            default=None,
            help="Write successful structured task output to a JSON file.",
        )
        task_run_parser.add_argument(
            "--pdf",
            dest="task_pdf",
            type=str,
            default=None,
            help="Attach one top-level PDF file input.",
        )
        task_enqueue_parser = task_command_parsers.add_parser(
            name="enqueue",
            description="Enqueue an intelligence task run",
            parents=[global_parser, task_input_parser, task_tool_parser],
        )
        task_enqueue_parser.add_argument(
            "definition",
            type=str,
            help="Task definition TOML file to enqueue",
        )
        task_enqueue_parser.add_argument(
            "--store-dsn",
            type=str,
            default=None,
            help="Durable task store PostgreSQL DSN.",
        )
        task_enqueue_parser.add_argument(
            "--store-schema",
            type=str,
            default=None,
            help="Durable task store PostgreSQL schema.",
        )
        task_enqueue_parser.add_argument(
            "--queue",
            type=str,
            default=None,
            help="Queue name to submit the task run to.",
        )
        task_enqueue_parser.add_argument(
            "--wait",
            action="store_true",
            help="Wait for the enqueued task to finish.",
        )
        task_enqueue_parser.add_argument(
            "--wait-timeout",
            type=float,
            default=None,
            help="Maximum seconds to wait for completion.",
        )
        task_enqueue_parser.add_argument(
            "--poll-interval",
            type=float,
            default=1.0,
            help="Seconds between wait polling attempts.",
        )
        task_enqueue_parser.add_argument(
            "--ephemeral",
            action="store_true",
            help="Reject non-durable storage for queued task runs.",
        )
        task_inspect_parser = task_command_parsers.add_parser(
            name="inspect",
            description="Inspect a task run",
            parents=[global_parser],
        )
        task_inspect_parser.add_argument(
            "run_id",
            type=str,
            help="Task run id to inspect",
        )
        task_inspect_parser.add_argument(
            "--store-dsn",
            type=str,
            default=None,
            help="Durable task store PostgreSQL DSN.",
        )
        task_inspect_parser.add_argument(
            "--store-schema",
            type=str,
            default=None,
            help="Durable task store PostgreSQL schema.",
        )
        task_inspect_parser.add_argument(
            "--after-sequence",
            type=int,
            default=None,
            help="Only include events after this sequence.",
        )
        task_usage_parser = task_command_parsers.add_parser(
            name="usage",
            description="Inspect task run usage",
            parents=[global_parser],
        )
        task_usage_parser.add_argument(
            "run_id",
            type=str,
            help="Task run id to inspect",
        )
        task_usage_parser.add_argument(
            "--store-dsn",
            type=str,
            default=None,
            help="Durable task store PostgreSQL DSN.",
        )
        task_usage_parser.add_argument(
            "--store-schema",
            type=str,
            default=None,
            help="Durable task store PostgreSQL schema.",
        )
        task_usage_parser.add_argument(
            "--attempt-id",
            type=str,
            default=None,
            help="Only include usage records for this attempt.",
        )
        task_usage_parser.add_argument(
            "--source",
            choices=("exact", "estimated", "unavailable"),
            default=None,
            help="Only include usage records from this source.",
        )
        task_output_parser = task_command_parsers.add_parser(
            name="output",
            description="Inspect a task run output",
            parents=[global_parser],
        )
        task_output_parser.add_argument(
            "run_id",
            type=str,
            help="Task run id to inspect",
        )
        task_output_parser.add_argument(
            "--store-dsn",
            type=str,
            default=None,
            help="Durable task store PostgreSQL DSN.",
        )
        task_output_parser.add_argument(
            "--store-schema",
            type=str,
            default=None,
            help="Durable task store PostgreSQL schema.",
        )
        task_events_parser = task_command_parsers.add_parser(
            name="events",
            description="Inspect task run events",
            parents=[global_parser],
        )
        task_events_parser.add_argument(
            "run_id",
            type=str,
            help="Task run id to inspect",
        )
        task_events_parser.add_argument(
            "--store-dsn",
            type=str,
            default=None,
            help="Durable task store PostgreSQL DSN.",
        )
        task_events_parser.add_argument(
            "--store-schema",
            type=str,
            default=None,
            help="Durable task store PostgreSQL schema.",
        )
        task_events_parser.add_argument(
            "--attempt-id",
            type=str,
            default=None,
            help="Only include events for this attempt.",
        )
        task_events_parser.add_argument(
            "--after-sequence",
            type=int,
            default=None,
            help="Only include events after this sequence.",
        )
        task_artifacts_parser = task_command_parsers.add_parser(
            name="artifacts",
            description="Inspect task run artifacts",
            parents=[global_parser],
        )
        task_artifacts_parser.add_argument(
            "run_id",
            type=str,
            help="Task run id to inspect",
        )
        task_artifacts_parser.add_argument(
            "--store-dsn",
            type=str,
            default=None,
            help="Durable task store PostgreSQL DSN.",
        )
        task_artifacts_parser.add_argument(
            "--store-schema",
            type=str,
            default=None,
            help="Durable task store PostgreSQL schema.",
        )
        task_worker_parser = task_command_parsers.add_parser(
            name="worker",
            description="Run a task queue worker",
            parents=[global_parser, task_tool_parser],
        )
        task_worker_parser.add_argument(
            "--queue",
            type=str,
            default="default",
            help="Task queue name to claim from.",
        )
        task_worker_parser.add_argument(
            "--store-dsn",
            type=str,
            default=None,
            help="Durable task store PostgreSQL DSN.",
        )
        task_worker_parser.add_argument(
            "--store-schema",
            type=str,
            default=None,
            help="Durable task store PostgreSQL schema.",
        )
        task_worker_parser.add_argument(
            "--worker-id",
            type=str,
            default=None,
            help="Stable worker id for queue claims.",
        )
        task_worker_parser.add_argument(
            "--once",
            action="store_true",
            help="Process at most one available task.",
        )
        task_worker_parser.add_argument(
            "--limit",
            type=int,
            default=1,
            help="Maximum task runs to process.",
        )
        task_worker_parser.add_argument(
            "--lease-seconds",
            type=int,
            default=300,
            help="Claim lease duration in seconds.",
        )
        task_worker_parser.add_argument(
            "--heartbeat-seconds",
            type=float,
            default=None,
            help="Optional worker heartbeat interval in seconds.",
        )
        task_worker_parser.add_argument(
            "--ephemeral",
            action="store_true",
            help="Reject non-durable storage for task workers.",
        )
        task_retention_sweep_parser = task_command_parsers.add_parser(
            name="retention-sweep",
            description="Delete expired task artifact bytes",
            parents=[global_parser],
        )
        task_retention_sweep_parser.add_argument(
            "--store-dsn",
            type=str,
            default=None,
            help="Durable task store PostgreSQL DSN.",
        )
        task_retention_sweep_parser.add_argument(
            "--store-schema",
            type=str,
            default=None,
            help="Durable task store PostgreSQL schema.",
        )
        task_retention_sweep_parser.add_argument(
            "--purpose",
            action="append",
            choices=("input", "output", "converted", "intermediate"),
            default=None,
            help="Artifact purpose to sweep. Can be passed more than once.",
        )
        task_retention_sweep_parser.add_argument(
            "--limit",
            type=int,
            default=100,
            help="Maximum expired artifacts to process.",
        )
        task_pgsql_parser = task_command_parsers.add_parser(
            name="pgsql",
            description="Manage task PostgreSQL schema migrations",
            parents=[global_parser],
        )
        task_pgsql_command_parsers = task_pgsql_parser.add_subparsers(
            dest="task_pgsql_command"
        )
        task_pgsql_common_parser = ArgumentParser(add_help=False)
        task_pgsql_common_parser.add_argument(
            "--dsn",
            type=str,
            default=None,
            help=(
                "PostgreSQL DSN. Defaults to AVALAN_TASK_PGSQL_DSN when "
                "omitted."
            ),
        )
        task_pgsql_common_parser.add_argument(
            "--schema",
            type=str,
            default=None,
            help=(
                "PostgreSQL schema. Defaults to AVALAN_TASK_PGSQL_SCHEMA "
                "when omitted."
            ),
        )
        task_pgsql_command_parsers.add_parser(
            name="status",
            description="Show the current task PostgreSQL schema revision",
            parents=[global_parser, task_pgsql_common_parser],
        )
        task_pgsql_migrate_parser = task_pgsql_command_parsers.add_parser(
            name="migrate",
            description="Apply task PostgreSQL schema migrations",
            parents=[global_parser, task_pgsql_common_parser],
        )
        task_pgsql_migrate_parser.add_argument(
            "migration_revision",
            nargs="?",
            type=str,
            default="head",
            help="Migration revision to apply.",
        )
        task_pgsql_command_parsers.add_parser(
            name="check",
            description="Check task PostgreSQL schema migration status",
            parents=[global_parser, task_pgsql_common_parser],
        )
        task_pgsql_stamp_parser = task_pgsql_command_parsers.add_parser(
            name="stamp",
            description="Stamp the task PostgreSQL schema revision",
            parents=[global_parser, task_pgsql_common_parser],
        )
        task_pgsql_stamp_parser.add_argument(
            "migration_revision",
            nargs="?",
            type=str,
            default="head",
            help="Migration revision to stamp.",
        )
        task_pgsql_command_parsers.add_parser(
            name="diagnose",
            description="Print safe task PostgreSQL migration diagnostics",
            parents=[global_parser, task_pgsql_common_parser],
        )

        # Memory command
        memory_parser = command_parsers.add_parser(
            name="memory", description="Manage memory", parents=[global_parser]
        )
        memory_command_parsers = memory_parser.add_subparsers(
            dest="memory_command"
        )
        memory_embeddings_parser = memory_command_parsers.add_parser(
            name="embeddings",
            description="Obtain and manipulate embeddings",
            parents=[
                global_parser,
                model_options_parser,
                memory_partitions_parser,
            ],
        )
        memory_embeddings_parser.add_argument(
            "--compare",
            type=str,
            action="append",
            help="If specified, compare embeddings with this string",
        )
        memory_embeddings_parser.add_argument(
            "--search",
            type=str,
            action="append",
            help="If specified, search across embeddings for this string",
        )
        memory_embeddings_parser.add_argument(
            "--search-k",
            default=1,
            type=int,
            help="How many nearest neighbors to obtain with search",
        )
        memory_embeddings_parser.add_argument(
            "--sort",
            type=DistanceType,
            choices=list(DistanceType),
            default=DistanceType.L2,
            help="Sort comparison results using the given similarity measure",
        )

        memory_search_parser = memory_command_parsers.add_parser(
            name="search",
            description="Search memories",
            parents=[
                global_parser,
                model_options_parser,
                memory_partitions_parser,
            ],
        )
        memory_search_parser.add_argument(
            "--dsn",
            type=str,
            required=True,
            help="PostgreSQL DSN for searching",
        )
        memory_search_parser.add_argument(
            "--participant",
            type=str,
            required=True,
            help="Participant ID to search",
        )
        memory_search_parser.add_argument(
            "--namespace", type=str, required=True, help="Namespace to search"
        )
        memory_search_parser.add_argument(
            "--function",
            type=VectorFunction,
            choices=list(VectorFunction),
            required=True,
            default=VectorFunction.L2_DISTANCE,
            help="Vector function to use for searching",
        )
        memory_search_parser.add_argument(
            "--limit", type=int, help="Return up to this many memories"
        )
        memory_doc_parser = memory_command_parsers.add_parser(
            name="document",
            description="Manage memory indexed documents",
        )
        memory_doc_command_parsers = memory_doc_parser.add_subparsers(
            dest="memory_document_command"
        )
        memory_doc_index_parser = memory_doc_command_parsers.add_parser(
            name="index",
            description="Add a document to the memory index",
            parents=[
                global_parser,
                model_options_parser,
                memory_partitions_parser,
            ],
        )
        memory_doc_index_parser.add_argument(
            "source",
            type=str,
            help="Source to index (an URL or a file path)",
        )
        memory_doc_index_parser.add_argument(
            "--partitioner",
            choices=["text", "code"],
            default="text",
            help="Partitioner to use when indexing a file",
        )
        memory_doc_index_parser.add_argument(
            "--language",
            type=str,
            help="Programming language for the code partitioner",
        )
        memory_doc_index_parser.add_argument(
            "--encoding",
            type=str,
            default="utf8",
            help="File encoding used when reading a local file",
        )
        memory_doc_index_parser.add_argument(
            "--identifier",
            type=str,
            help="Identifier for the memory entry (defaults to the source)",
        )
        memory_doc_index_parser.add_argument(
            "--title",
            type=str,
            help="Title for the memory entry",
        )
        memory_doc_index_parser.add_argument(
            "--description",
            type=str,
            help="Description for the memory entry",
        )
        memory_doc_index_parser.add_argument(
            "--dsn",
            type=str,
            required=True,
            help="PostgreSQL DSN for storing the document",
        )
        memory_doc_index_parser.add_argument(
            "--participant",
            type=str,
            required=True,
            help="Participant ID for the memory entry",
        )
        memory_doc_index_parser.add_argument(
            "--namespace",
            type=str,
            required=True,
            help="Namespace for the memory entry",
        )

        # Model command
        model_parser = command_parsers.add_parser(
            name="model",
            description=(
                "Manage a model, showing details, loading or downloading it"
            ),
        )
        model_command_parsers = model_parser.add_subparsers(
            dest="model_command"
        )

        model_display_parser = model_command_parsers.add_parser(
            name="display",
            description="Show information about a model",
            parents=[global_parser, model_options_parser],
        )
        model_display_parser.add_argument(
            "--sentence-transformer",
            help="Load the model as a SentenceTransformer model",
            default=False,
            action="store_true",
        )
        model_display_parser.add_argument(
            "--summary",
            default=False,
            action="store_true",
        )

        model_command_parsers.add_parser(
            name="install",
            description="Install a model",
            parents=[global_parser, model_install_parser],
        )
        model_run_parser = model_command_parsers.add_parser(
            name="run",
            description="Run a model",
            parents=[
                global_parser,
                model_options_parser,
                model_inference_display_parser,
            ],
        )
        model_run_parser.add_argument(
            "--attention",
            type=str,
            choices=get_args(AttentionImplementation),
            default=default_attention,
            help=(
                "Attention implementation to use "
                f"(defaults to best available: {default_attention})"
            ),
        )
        model_run_parser.add_argument(
            "--output-hidden-states",
            action="store_true",
            default=False,
            help="Return hidden states for each layer",
        )
        model_run_parser.add_argument(
            "--path",
            type=str,
            help=(
                "Path where to store generated audio or vision output. "
                "Only applicable to audio and vision modalities."
            ),
        )
        model_run_parser.add_argument(
            "--checkpoint",
            type=str,
            help=(
                "AnimateDiff motion adapter checkpoint to use. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--base-model",
            type=str,
            help=(
                "ID of the base model for text-to-video generation. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--upsampler-model",
            type=str,
            help=(
                "Upsampler model to use for text-to-video generation. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--refiner-model",
            type=str,
            help=(
                "Expert model to use for refinement. "
                "Only applicable to vision text to image modality."
            ),
        )
        model_run_parser.add_argument(
            "--audio-reference-path",
            type=str,
            help=(
                "Path to existing audio file to use for voice cloning. "
                "Only applicable to audio modalities."
            ),
        )
        model_run_parser.add_argument(
            "--audio-reference-text",
            type=str,
            help=(
                "Text transcript of the reference audio given in "
                "--audio-reference-path. "
                "Only applicable to audio modalities."
            ),
        )
        model_run_parser.add_argument(
            "--audio-sampling-rate",
            default=44_100,
            type=int,
            help=(
                "Sampling rate to use for audio generation. "
                "Only applicable to audio modalities."
            ),
        )
        model_run_parser.add_argument(
            "--vision-threshold",
            dest="vision_threshold",
            default=0.3,
            type=float,
            help=(
                "Score threshold for object detection. "
                "Only applicable to vision modalities."
            ),
        )
        model_run_parser.add_argument(
            "--vision-width",
            dest="vision_width",
            type=int,
            help=(
                "Resize input image to this width before processing. "
                "Only applicable to vision image text to text modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-color-model",
            default=VisionColorModel.RGB,
            type=str,
            choices=[m.value for m in VisionColorModel],
            help=(
                "Color model for image generation. "
                "Only applicable to vision text to image modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-image-format",
            default=VisionImageFormat.JPEG,
            type=str,
            choices=[f.value for f in VisionImageFormat],
            help=(
                "Image format to save generated image. "
                "Only applicable to vision text to image modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-high-noise-frac",
            dest="vision_high_noise_frac",
            default=0.8,
            type=float,
            help=(
                "High noise fraction for diffusion (controls the split "
                "point between the base model and the refiner. "
                "Only applicable to vision text to image modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-steps",
            dest="vision_steps",
            default=150,
            type=int,
            help=(
                "Number of denoising (sampling) iterations in the "
                "diffusion scheduler. "
                "Only applicable to vision text to image modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-timestep-spacing",
            default=TimestepSpacing.TRAILING,
            type=str,
            choices=[t.value for t in TimestepSpacing],
            help=(
                "Timestep spacing strategy for the Euler scheduler. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-beta-schedule",
            default=BetaSchedule.LINEAR,
            type=str,
            choices=[b.value for b in BetaSchedule],
            help=(
                "Beta schedule for the Euler scheduler. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-guidance-scale",
            default=1.0,
            type=float,
            help=(
                "Guidance scale for text-to-video generation. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-reference-path",
            type=str,
            help=(
                "Reference image to guide generation. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-negative-prompt",
            type=str,
            help=(
                "Negative prompt for generation. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-height",
            type=int,
            help=(
                "Height of generated video. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-downscale",
            default=2 / 3,
            type=float,
            help=(
                "Downscale factor for upsampling. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-frames",
            default=96,
            type=int,
            help=(
                "Number of frames to generate. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-denoise-strength",
            default=0.4,
            type=float,
            help=(
                "Denoise strength for upsampling. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-inference-steps",
            default=10,
            type=int,
            help=(
                "Number of inference steps for upsampler. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-decode-timestep",
            default=0.05,
            type=float,
            help=(
                "Decode timestep for video decoding. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-noise-scale",
            default=0.025,
            type=float,
            help=(
                "Noise scale for video generation. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--vision-fps",
            default=24,
            type=int,
            help=(
                "Frames per second for generated video. "
                "Only applicable to vision text to video modality."
            ),
        )
        model_run_parser.add_argument(
            "--do-sample",
            default=True,
            action="store_true",
            help=(
                "Tell if the token generation process should be "
                "deterministic or stochastic. When enabled, it's stochastic "
                "and uses probability distribution."
            ),
        )
        model_run_parser.add_argument(
            "--enable-gradient-calculation",
            default=False,
            action="store_true",
            help="Enable gradient calculation.",
        )
        model_run_parser.add_argument(
            "--use-cache",
            default=False,
            action="store_true",
            help=(
                "Past key values are used to speed up decoding if applicable "
                "to model."
            ),
        )
        model_run_parser.add_argument(
            "--max-new-tokens",
            default=10 * 1024,
            type=int,
            help="Maximum number of tokens to generate",
        )
        model_run_parser.add_argument(
            "--openai-response-failed-retries",
            type=CLI._non_negative_integer,
            default=None,
            help=(
                "OpenAI Responses empty response.failed stream retries. "
                "Set to 0 to disable."
            ),
        )
        model_run_parser.add_argument(
            "--openai-response-failed-retry-delay-seconds",
            type=CLI._non_negative_number,
            default=None,
            help=(
                "OpenAI Responses empty response.failed retry delay "
                "in seconds."
            ),
        )
        model_run_parser.add_argument(
            "--modality",
            default=Modality.TEXT_GENERATION,
            type=str,
            choices=[m.value for m in Modality],
        )
        model_run_parser.add_argument(
            "--min-p",
            type=float,
            help=(
                "Minimum token probability, which will be scaled by the "
                "probability of the most likely token [0, 1]"
            ),
        )
        model_run_parser.add_argument(
            "--repetition-penalty",
            default=1.0,
            type=float,
            help=(
                "Exponential penalty on sequences not in the original input. "
                "Defaults to 1.0, which means no penalty."
            ),
        )
        model_run_parser.add_argument(
            "--skip-special-tokens",
            default=False,
            action="store_true",
            help="If specified, skip special tokens when decoding",
        )
        model_run_parser.add_argument(
            "--system",
            type=str,
            help="Use this as system prompt",
        )
        model_run_parser.add_argument(
            "--developer",
            type=str,
            help="Use this as developer prompt",
        )
        model_run_parser.add_argument(
            "--input-file",
            action="append",
            help=(
                "Attach a local file as native input for text generation. "
                "May be specified multiple times."
            ),
        )
        CLI._add_ds4_backend_options(model_run_parser)
        model_run_parser.add_argument(
            "--text-context",
            type=str,
            help="Context string for question answering",
        )
        model_run_parser.add_argument(
            "--text-labeled-only",
            default=None,
            action="store_true",
            help=(
                "If specified, only tokens with labels detected are "
                "returned. "
                "Only applicable to text_token_classification modalities."
            ),
        )
        model_run_parser.add_argument(
            "--text-max-length",
            type=int,
            help=(
                "The maximum length the generated tokens can have. Corresponds"
                " to the length of the input prompt + max_new_tokens"
            ),
        )
        model_run_parser.add_argument(
            "--text-num-beams",
            type=int,
            default=1,
            help="Number of beams for beam search. 1 means no beam search",
        )
        model_run_parser.add_argument(
            "--text-disable-cache",
            dest="use_cache",
            action="store_false",
            help="If specified, disable generation cache",
        )
        model_run_parser.add_argument(
            "--text-cache-strategy",
            type=str,
            choices=[c.value for c in GenerationCacheStrategy],
            dest="cache_strategy",
            help="Cache implementation to use for generation",
        )
        model_run_parser.add_argument(
            "--text-from-lang",
            type=str,
            help="Source language code for text translation",
        )
        model_run_parser.add_argument(
            "--text-to-lang",
            type=str,
            help="Destination language code for text translation",
        )
        model_run_parser.add_argument(
            "--start-thinking",
            default=False,
            action="store_true",
            help="If specified, assume model response starts with reasoning",
        )
        model_run_parser.add_argument(
            "--chat-disable-thinking",
            dest="chat_disable_thinking",
            action="store_true",
            default=False,
            help="Disable thinking tokens in chat template",
        )
        model_run_parser.add_argument(
            "--no-reasoning",
            action="store_true",
            default=False,
            help="Disable reasoning parser",
        )
        model_run_parser.add_argument(
            "--reasoning-tag",
            type=str,
            choices=[t.value for t in ReasoningTag],
            help="Reasoning tag style",
        )
        model_run_parser.add_argument(
            "--reasoning-effort",
            type=str,
            choices=[e.value for e in ReasoningEffort],
            help="Reasoning effort level",
        )
        model_run_parser.add_argument(
            "--reasoning-max-new-tokens",
            type=int,
            help="Maximum number of reasoning tokens",
        )
        model_run_parser.add_argument(
            "--reasoning-stop-on-max-new-tokens",
            action="store_true",
            default=False,
            help="Stop reasoning when maximum tokens are produced",
        )
        model_run_parser.add_argument(
            "--stop_on_keyword",
            type=str,
            action="append",
            help="Stop token generation when this keyword is found",
        )
        model_run_parser.add_argument(
            "--temperature",
            default=0.7,
            type=float,
            help="Temperature [0, 1]",
        )
        model_run_parser.add_argument(
            "--top-k",
            type=int,
            help=(
                "Number of highest probability vocabulary tokens to keep for "
                "top-k-filtering."
            ),
        )
        model_run_parser.add_argument(
            "--top-p",
            type=float,
            help=(
                "If set to < 1, only the smallest set of most probable "
                "tokens with probabilities that add up to top_p or higher "
                "are kept for generation."
            ),
        )
        model_run_parser.add_argument(
            "--trust-remote-code",
            action="store_true",
        )
        model_search_parser = model_command_parsers.add_parser(
            name="search",
            description="Search for models",
            parents=[global_parser],
        )
        model_search_parser.add_argument(
            "--search",
            type=str,
            action="append",
            required=False,
            help="Search for models matching given expression",
        )
        model_search_parser.add_argument(
            "--filter",
            type=str,
            action="append",
            help="Filter models on this (e.g: text-classification)",
        )
        model_search_parser.add_argument(
            "--library",
            type=str,
            action="append",
            help="Filter by library",
        )
        model_search_parser.add_argument(
            "--author",
            type=str,
            help="Filter by author",
        )
        gated_group = model_search_parser.add_mutually_exclusive_group()
        gated_group.add_argument(
            "--gated",
            action="store_true",
            help="Only gated models",
        )
        gated_group.add_argument(
            "--open",
            action="store_true",
            help="Only open models",
        )
        model_search_parser.add_argument(
            "--language",
            type=str,
            action="append",
            help="Filter by language",
        )
        model_search_parser.add_argument(
            "--name",
            type=str,
            action="append",
            help="Filter by model name",
        )
        model_search_parser.add_argument(
            "--task",
            type=str,
            action="append",
            help="Filter by task",
        )
        model_search_parser.add_argument(
            "--tag",
            type=str,
            action="append",
            help="Filter by tag",
        )
        model_search_parser.add_argument(
            "--limit",
            default=10,
            type=int,
            help="Maximum number of models to return",
        )
        model_uninstall_parser = model_command_parsers.add_parser(
            name="uninstall",
            description="Uninstall a model",
            parents=[global_parser, model_options_parser],
        )
        model_uninstall_parser.add_argument(
            "--delete",
            action="store_true",
            help=(
                "Actually delete. If not provided, a dry run is performed "
                "and data that would be deleted is shown, yet not deleted"
            ),
        )

        # Tokenizer command
        tokenizer_parser = command_parsers.add_parser(
            name="tokenizer",
            description=(
                "Manage tokenizers, loading, modifying and saving them"
            ),
            parents=[global_parser],
        )
        tokenizer_parser.add_argument(
            "--tokenizer",
            "-t",
            type=str,
            required=True,
            help="Tokenizer to load",
        )
        tokenizer_parser.add_argument(
            "--save",
            type=str,
            help=(
                "Save tokenizer (useful if modified via --special-token or "
                "--token) to given path, only if model is loaded"
            ),
        )
        tokenizer_parser.add_argument(
            "--special-token",
            type=str,
            action="append",
            help="Special token to add to tokenizer",
        )
        tokenizer_parser.add_argument(
            "--token",
            type=str,
            action="append",
            help="Token to add to tokenizer",
        )

        # Train command
        train_parser = command_parsers.add_parser(
            name="train", description="Training", parents=[global_parser]
        )
        train_command_parsers = train_parser.add_subparsers(
            dest="train_command"
        )
        train_run_parser = train_command_parsers.add_parser(
            name="run",
            description="Run a given training",
            parents=[global_parser],
        )
        train_run_parser.add_argument(
            "training",
            type=str,
            help="Training to run",
        )

        parser.add_argument(
            "--help-full",
            action="store_true",
            help="Show help for all commands and subcommands",
        )

        return parser

    @staticmethod
    def _get_translator(
        app_name: str, locales_path: str, locale: str
    ) -> Translator:
        """Return translation object for ``locale`` or ``gettext`` fallback."""
        try:
            return translation(
                app_name, localedir=locales_path, languages=[locale]
            )
        except FileNotFoundError:
            return gettext

    @staticmethod
    def _extract_chat_settings(
        argv: list[str],
    ) -> tuple[list[str], dict[str, bool]]:
        """Return ``argv`` without chat options and extracted flags."""
        options: dict[str, bool] = {}
        new_argv: list[str] = []
        for arg in argv:
            if arg.startswith("--run-chat-"):
                key = arg[len("--run-chat-") :].replace("-", "_")
                options[key] = True
            else:
                new_argv.append(arg)
        return new_argv, options

    @staticmethod
    def _add_agent_server_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add shared server options for agent commands."""
        parser.add_argument(
            "--host",
            default="127.0.0.1",
            type=str,
            help="Host (defaults to 127.0.0.1)",
        )
        parser.add_argument(
            "--port",
            default=9001,
            type=int,
            help="Port (defaults to 9001, HAL 9000+1)",
        )
        parser.add_argument(
            "--mcp-prefix",
            default="/mcp",
            type=str,
            help="URL prefix for MCP endpoints (defaults to /mcp)",
        )
        parser.add_argument(
            "--mcp-name",
            default="run",
            type=str,
            help="MCP tool name for tools/call (defaults to run)",
        )
        parser.add_argument(
            "--mcp-description",
            type=str,
            help="MCP tool description for tools/list",
        )
        parser.add_argument(
            "--openai-prefix",
            default="/v1",
            type=str,
            help="URL prefix for OpenAI endpoints (defaults to /v1)",
        )
        parser.add_argument(
            "--a2a-prefix",
            default="/a2a",
            type=str,
            help="URL prefix for A2A endpoints (defaults to /a2a)",
        )
        parser.add_argument(
            "--a2a-name",
            default="run",
            type=str,
            help="A2A tool name for task execution (defaults to run)",
        )
        parser.add_argument(
            "--a2a-description",
            type=str,
            help="A2A tool description for the agent card",
        )
        parser.add_argument(
            "--protocol",
            action="append",
            dest="protocol",
            help=(
                "Protocol to expose (e.g. openai,"
                " openai:responses,completion). May be specified multiple"
                " times"
            ),
        )
        parser.add_argument(
            "--reload",
            action="store_true",
            default=False,
            help="Hot reload on code changes",
        )
        parser.add_argument(
            "--cors-origin",
            action="append",
            help="Allowed CORS origin; may be specified multiple times",
        )
        parser.add_argument(
            "--cors-origin-regex",
            type=str,
            help="Allowed CORS origin regex",
        )
        parser.add_argument(
            "--cors-method",
            action="append",
            help="Allowed CORS method; may be specified multiple times",
        )
        parser.add_argument(
            "--cors-header",
            action="append",
            help="Allowed CORS header; may be specified multiple times",
        )
        parser.add_argument(
            "--cors-credentials",
            action="store_true",
            default=False,
            help="Allow CORS credentials",
        )
        parser.add_argument(
            "--server-output-redaction",
            dest="server_output_redaction_enabled",
            action="store_true",
            default=None,
            help="Enable opt-in redaction for server protocol output.",
        )
        parser.add_argument(
            "--server-output-redaction-rule",
            dest="server_output_redaction_rules",
            action="append",
            choices=SERVER_OUTPUT_REDACTION_RULES,
            default=None,
            help=(
                "Enable server output redaction and restrict it to a rule;"
                " may be specified multiple times."
            ),
        )
        parser.add_argument(
            "--server-output-redaction-protocol",
            dest="server_output_redaction_protocols",
            action="append",
            choices=SERVER_OUTPUT_REDACTION_PROTOCOLS,
            default=None,
            help=(
                "Enable server output redaction and restrict it to a"
                " protocol; may be specified multiple times."
            ),
        )
        parser.add_argument(
            "--server-output-redaction-channel",
            dest="server_output_redaction_channels",
            action="append",
            choices=SERVER_OUTPUT_REDACTION_CHANNELS,
            default=None,
            help=(
                "Enable server output redaction and restrict it to a"
                " channel; may be specified multiple times."
            ),
        )
        return parser

    @staticmethod
    def _add_ds4_backend_options(parser: ArgumentParser) -> _ArgumentGroup:
        group = parser.add_argument_group("DS4 backend options")
        group.add_argument(
            "--ds4-ctx",
            dest="ds4_ctx",
            type=int,
            default=None,
            help="DS4 context size",
        )
        group.add_argument(
            "--ds4-native-backend",
            dest="ds4_native_backend",
            type=str,
            choices=["auto", "metal", "cuda", "cpu"],
            default=None,
            help="DS4 native backend",
        )
        group.add_argument(
            "--ds4-mtp",
            dest="ds4_mtp",
            type=str,
            default=None,
            help="DS4 MTP model path",
        )
        group.add_argument(
            "--ds4-mtp-draft",
            dest="ds4_mtp_draft",
            type=int,
            default=None,
            help="DS4 MTP draft-token count",
        )
        group.add_argument(
            "--ds4-mtp-margin",
            dest="ds4_mtp_margin",
            type=float,
            default=None,
            help="DS4 MTP acceptance margin",
        )
        group.add_argument(
            "--ds4-warm-weights",
            dest="ds4_warm_weights",
            action="store_true",
            default=None,
            help="Warm DS4 model weights when opening the engine",
        )
        group.add_argument(
            "--ds4-quality",
            dest="ds4_quality",
            action="store_true",
            default=None,
            help="Enable DS4 quality mode",
        )
        group.add_argument(
            "--with-ds4-native-log",
            "--ds4-native-log",
            dest="ds4_native_log",
            action="store_true",
            default=None,
            help="Replay DS4 native stderr emitted while opening the engine",
        )
        group.add_argument(
            "--no-ds4-native-log",
            dest="ds4_native_log",
            action="store_false",
            help="Suppress DS4 native stderr emitted while opening the engine",
        )
        return group

    @staticmethod
    def _add_agent_settings_arguments(
        parser: ArgumentParser,
    ) -> _ArgumentGroup:
        group = parser.add_argument_group("inline agent settings")
        group.add_argument("--engine-uri", type=str, help="Agent engine URI")
        group.add_argument(
            "--engine-base-url",
            type=str,
            help="Agent engine provider base URL",
        )
        group.add_argument("--name", type=str, help="Agent name")
        group.add_argument("--role", type=str, help="Agent role")
        group.add_argument("--task", type=str, help="Agent task")
        group.add_argument(
            "--instructions",
            type=str,
            help="Provider instructions",
        )
        group.add_argument(
            "--goal-instructions",
            type=str,
            help="Agent goal instructions",
        )
        group.add_argument("--system", type=str, help="System prompt")
        group.add_argument("--developer", type=str, help="Developer prompt")
        group.add_argument("--user", type=str, help="User message template")
        group.add_argument(
            "--user-template", type=str, help="User message template file"
        )
        group.add_argument(
            "--memory-recent",
            dest="memory_recent",
            action="store_true",
            default=None,
        )
        group.add_argument(
            "--no-memory-recent", dest="memory_recent", action="store_false"
        )
        group.add_argument(
            "--memory-permanent-message",
            type=str,
            help="Permanent message memory DSN",
        )
        group.add_argument(
            "--memory-permanent",
            action="append",
            dest="memory_permanent",
            help="Permanent memory definition namespace@dsn",
        )
        group.add_argument(
            "--memory-engine-model-id",
            type=str,
            default=_DEFAULT_SENTENCE_MODEL_ID,
            help="Sentence transformer model for memory",
        )
        group.add_argument(
            "--memory-engine-max-tokens",
            type=int,
            default=_DEFAULT_SENTENCE_MODEL_MAX_TOKENS,
            help="Maximum tokens for memory sentence transformer",
        )
        group.add_argument(
            "--memory-engine-overlap",
            type=int,
            default=_DEFAULT_SENTENCE_MODEL_OVERLAP_SIZE,
            help="Overlap size for memory sentence transformer",
        )
        group.add_argument(
            "--memory-engine-window",
            type=int,
            default=_DEFAULT_SENTENCE_MODEL_WINDOW_SIZE,
            help="Window size for memory sentence transformer",
        )
        group.add_argument(
            "--run-max-new-tokens",
            type=int,
            help="Maximum count of tokens on output",
            default=None,
        )
        group.add_argument(
            "--maximum-tool-cycles",
            "--run-maximum-tool-cycles",
            type=CLI._maximum_tool_cycles,
            dest="run_maximum_tool_cycles",
            help=(
                "Maximum model/tool result cycles for an agent run, "
                "or 'unlimited'"
            ),
            default=None,
        )
        group.add_argument(
            "--block-repeated-tool-calls",
            "--run-block-repeated-tool-calls",
            action="store_true",
            default=None,
            dest="run_block_repeated_tool_calls",
            help="Stop repeated same-name/same-argument tool calls.",
        )
        group.add_argument(
            "--run-skip-special-tokens",
            action="store_true",
            default=False,
            help="Skip special tokens on output",
        )
        group.add_argument(
            "--run-disable-cache",
            dest="run_use_cache",
            action="store_false",
            default=None,
            help="Disable generation cache",
        )
        group.add_argument(
            "--run-cache-strategy",
            type=str,
            choices=[c.value for c in GenerationCacheStrategy],
            dest="run_cache_strategy",
            help="Cache implementation to use for generation",
        )
        group.add_argument(
            "--run-openai-response-failed-retries",
            type=CLI._non_negative_integer,
            default=None,
            help=(
                "OpenAI Responses empty response.failed stream retries. "
                "Set to 0 to disable."
            ),
        )
        group.add_argument(
            "--run-openai-response-failed-retry-delay-seconds",
            type=CLI._non_negative_number,
            default=None,
            help=(
                "OpenAI Responses empty response.failed retry delay "
                "in seconds."
            ),
        )
        group.add_argument(
            "--run-temperature",
            default=None,
            type=float,
            help="Temperature [0, 1]",
        )
        group.add_argument(
            "--run-top-k",
            type=int,
            help=(
                "Number of highest probability vocabulary tokens to keep for "
                "top-k-filtering."
            ),
        )
        group.add_argument(
            "--run-top-p",
            type=float,
            help=(
                "If set to < 1, only the smallest set of most probable "
                "tokens with probabilities that add up to top_p or higher "
                "are kept for generation."
            ),
        )
        group.add_argument(
            "--reasoning-effort",
            "--run-reasoning-effort",
            type=str,
            choices=[e.value for e in ReasoningEffort],
            dest="run_reasoning_effort",
            help="Reasoning effort level",
        )
        group.add_argument(
            "--tool", type=str, action="append", help="Enable tool"
        )
        group.add_argument(
            "--tools",
            type=str,
            action="append",
            help="Enable tools matching namespace",
        )
        return group

    @staticmethod
    def _add_tool_settings_arguments(
        parser: ArgumentParser, *, prefix: str, settings_cls: type
    ) -> _ArgumentGroup:
        """Add dataclass based tool options to ``parser``."""
        group = parser.add_argument_group(f"{prefix} tool settings")
        scalar_fields = getattr(settings_cls, "CLI_SCALAR_FIELDS", None)

        for field in fields(settings_cls):
            if scalar_fields is not None and field.name not in scalar_fields:
                continue
            option = f"--tool-{prefix}-{field.name.replace('_', '-')}"
            dest = f"tool_{prefix}_{field.name}"

            ftype = field.type
            origin = get_origin(ftype)
            args = get_type_args(ftype)
            is_sequence = False
            if origin is not None and type(None) in args:
                ftype = next((a for a in args if a is not type(None)), str)
                origin = get_origin(ftype)
                args = get_type_args(ftype)
            if origin is not None:
                if origin is list or origin is tuple:
                    is_sequence = True
                    ftype = args[0]
                elif origin.__name__ == "Literal":
                    ftype = type(args[0])

            if (
                settings_cls is ShellToolSettings
                and prefix == "shell"
                and field.name in {"backend", "execution_mode"}
            ):
                group.add_argument(
                    option,
                    dest=dest,
                    choices=("local", "sandbox", "container"),
                    default=None,
                    help="Trusted shell execution mode.",
                )
            elif (
                settings_cls is ShellToolSettings
                and prefix == "shell"
                and field.name == "pipeline_transport"
            ):
                group.add_argument(
                    option,
                    dest=dest,
                    choices=("buffered", "native"),
                    default=None,
                    help="Trusted shell pipeline byte transport.",
                )
            elif ftype is bool or isinstance(field.default, bool):
                if (
                    settings_cls is ShellToolSettings
                    and prefix == "shell"
                    and field.name == "input_file_manifest_enabled"
                ):
                    group.add_argument(
                        option,
                        dest=dest,
                        action="store_true",
                        default=None,
                    )
                    group.add_argument(
                        f"--no-tool-{prefix}-{field.name.replace('_', '-')}",
                        dest=dest,
                        action="store_false",
                    )
                else:
                    group.add_argument(
                        option, dest=dest, action="store_true", default=None
                    )
            elif ftype is int or isinstance(field.default, int):
                group.add_argument(
                    option,
                    dest=dest,
                    action="append" if is_sequence else "store",
                    type=int,
                    default=None,
                )
            elif ftype is float or isinstance(field.default, float):
                group.add_argument(
                    option,
                    dest=dest,
                    action="append" if is_sequence else "store",
                    type=float,
                    default=None,
                )
            else:
                group.add_argument(
                    option,
                    dest=dest,
                    action="append" if is_sequence else "store",
                    type=str,
                    default=None,
                )

        if settings_cls is ShellToolSettings and prefix == "shell":
            CLI._add_shell_git_settings_arguments(group)
            group.add_argument(
                "--tool-shell-executable-search-path",
                dest="tool_shell_executable_search_paths",
                action="append",
                type=str,
                default=None,
                help="Add a trusted directory used to resolve shell tools.",
            )
            group.add_argument(
                "--tool-shell-executable-path",
                dest="tool_shell_executable_paths",
                action="append",
                type=CLI._parse_shell_executable_path,
                default=None,
                metavar="COMMAND=PATH",
                help="Map a shell command to a trusted absolute executable.",
            )
            group.add_argument(
                "--tool-container-backend",
                dest="tool_container_backend",
                choices=(
                    "docker",
                    "apple-container",
                ),
                default=None,
                help="Trusted container backend for shell tools.",
            )
            group.add_argument(
                "--tool-container-profile",
                dest="tool_container_profile",
                type=str,
                default=None,
                help="Trusted default container profile name.",
            )
            group.add_argument(
                "--tool-container-image",
                dest="tool_container_image",
                type=str,
                default=None,
                help="Digest-pinned image reference for the trusted profile.",
            )
            group.add_argument(
                "--tool-container-workspace-root",
                dest="tool_container_workspace_root",
                type=str,
                default=None,
                help="Trusted host workspace root mounted into the container.",
            )
            group.add_argument(
                "--tool-container-pull-policy",
                dest="tool_container_pull_policy",
                choices=("never", "if_missing", "always"),
                default=None,
                help="Trusted image pull policy.",
            )
            group.add_argument(
                "--tool-container-platform",
                dest="tool_container_platform",
                type=str,
                default=None,
                help="Trusted target image platform such as linux/amd64.",
            )
            group.add_argument(
                "--tool-container-cpu-count",
                dest="tool_container_cpu_count",
                type=int,
                default=None,
                help="Trusted default container CPU limit.",
            )
            group.add_argument(
                "--tool-container-memory-bytes",
                dest="tool_container_memory_bytes",
                type=int,
                default=None,
                help="Trusted default container memory limit in bytes.",
            )
            group.add_argument(
                "--tool-container-pids",
                dest="tool_container_pids",
                type=int,
                default=None,
                help="Trusted default container PID limit.",
            )
            group.add_argument(
                "--tool-container-timeout-seconds",
                dest="tool_container_timeout_seconds",
                type=int,
                default=None,
                help="Trusted default container execution timeout.",
            )
            group.add_argument(
                "--tool-container-network-mode",
                dest="tool_container_network_mode",
                choices=("none", "loopback", "allowlist", "full"),
                default=None,
                help="Trusted default container network mode.",
            )
            group.add_argument(
                "--tool-container-review-mode",
                dest="tool_container_review_mode",
                choices=("deny", "require_review", "preauthorized"),
                default=None,
                help="Trusted container escalation review mode.",
            )
            group.add_argument(
                "--tool-sandbox-backend",
                dest="tool_sandbox_backend",
                choices=("seatbelt", "bubblewrap"),
                default=None,
                help="Trusted sandbox backend for shell tools.",
            )
            group.add_argument(
                "--tool-sandbox-profile",
                dest="tool_sandbox_profile",
                type=str,
                default=None,
                help="Trusted default sandbox profile name.",
            )
            group.add_argument(
                "--tool-sandbox-trusted-executable",
                dest="tool_sandbox_trusted_executables",
                action="append",
                type=str,
                default=None,
                help="Trusted absolute executable allowed in the sandbox.",
            )
            group.add_argument(
                "--tool-sandbox-executable-search-root",
                dest="tool_sandbox_executable_search_roots",
                action="append",
                type=str,
                default=None,
                help="Trusted absolute executable search root.",
            )
            group.add_argument(
                "--tool-sandbox-read-root",
                dest="tool_sandbox_read_roots",
                action="append",
                type=str,
                default=None,
                help="Trusted absolute read root for the sandbox.",
            )
            group.add_argument(
                "--tool-sandbox-write-root",
                dest="tool_sandbox_write_roots",
                action="append",
                type=str,
                default=None,
                help="Trusted absolute write root for the sandbox.",
            )
            group.add_argument(
                "--tool-sandbox-deny-root",
                dest="tool_sandbox_deny_roots",
                action="append",
                type=str,
                default=None,
                help="Trusted absolute deny root for the sandbox.",
            )
            group.add_argument(
                "--tool-sandbox-scratch-root",
                dest="tool_sandbox_scratch_roots",
                action="append",
                type=str,
                default=None,
                help="Trusted absolute scratch root for sandbox temp files.",
            )
            group.add_argument(
                "--tool-sandbox-output-root",
                dest="tool_sandbox_output_roots",
                action="append",
                type=str,
                default=None,
                help="Trusted absolute output root for generated artifacts.",
            )
            group.add_argument(
                "--tool-sandbox-network-mode",
                dest="tool_sandbox_network_mode",
                choices=("none", "loopback", "allowlist", "full"),
                default=None,
                help="Trusted sandbox network mode.",
            )
            group.add_argument(
                "--tool-sandbox-network-egress",
                dest="tool_sandbox_network_egress",
                action="append",
                type=str,
                default=None,
                help="Trusted sandbox network egress allowlist entry.",
            )
            group.add_argument(
                "--tool-sandbox-timeout-seconds",
                dest="tool_sandbox_timeout_seconds",
                type=int,
                default=None,
                help="Trusted sandbox execution timeout.",
            )
            group.add_argument(
                "--tool-sandbox-pids",
                dest="tool_sandbox_pids",
                type=int,
                default=None,
                help="Trusted sandbox process limit.",
            )
            group.add_argument(
                "--tool-sandbox-max-stdout-bytes",
                dest="tool_sandbox_max_stdout_bytes",
                type=int,
                default=None,
                help="Trusted sandbox stdout byte limit.",
            )
            group.add_argument(
                "--tool-sandbox-max-stderr-bytes",
                dest="tool_sandbox_max_stderr_bytes",
                type=int,
                default=None,
                help="Trusted sandbox stderr byte limit.",
            )
            group.add_argument(
                "--tool-sandbox-max-artifact-bytes",
                dest="tool_sandbox_max_artifact_bytes",
                type=int,
                default=None,
                help="Trusted sandbox artifact byte limit.",
            )
            group.add_argument(
                "--tool-sandbox-allow-artifacts",
                dest="tool_sandbox_allow_artifacts",
                action="store_true",
                default=None,
                help="Allow sandbox generated artifacts.",
            )
            group.add_argument(
                "--tool-sandbox-child-processes",
                dest="tool_sandbox_child_processes",
                choices=("deny", "allow"),
                default=None,
                help="Trusted sandbox child process policy.",
            )
            group.add_argument(
                "--tool-sandbox-inherited-fds",
                dest="tool_sandbox_inherited_fds",
                choices=("deny", "stdio", "explicit"),
                default=None,
                help="Trusted sandbox inherited file descriptor policy.",
            )
            group.add_argument(
                "--tool-shell-container-profile",
                dest="tool_shell_container_profile",
                type=str,
                default=None,
                help="Select an approved container profile for shell tools.",
            )
            group.add_argument(
                "--tool-shell-container-required",
                dest="tool_shell_container_required",
                action="store_true",
                default=None,
                help="Fail closed instead of falling back to local shell.",
            )
            group.add_argument(
                "--tool-shell-sandbox-profile",
                dest="tool_shell_sandbox_profile",
                type=str,
                default=None,
                help="Select an approved sandbox profile for shell tools.",
            )
            group.add_argument(
                "--tool-shell-sandbox-required",
                dest="tool_shell_sandbox_required",
                action="store_true",
                default=None,
                help="Fail closed instead of falling back to local shell.",
            )

        return group

    @staticmethod
    def _add_shell_git_settings_arguments(group: _ArgumentGroup) -> None:
        git_scalar_fields = ShellGitToolSettings.CLI_SCALAR_FIELDS
        git_sequence_fields = ShellGitToolSettings.CLI_SEQUENCE_FIELDS
        git_fields = {
            dataclass_field.name: dataclass_field
            for dataclass_field in fields(ShellGitToolSettings)
        }
        for field_name in (*git_scalar_fields, *git_sequence_fields):
            field = git_fields[field_name]
            option = f"--tool-shell-git-{field_name.replace('_', '-')}"
            dest = f"tool_shell_git_{field_name}"
            if field_name in git_sequence_fields:
                group.add_argument(
                    option,
                    dest=dest,
                    action="append",
                    type=str,
                    default=None,
                    help="Trusted shell Git list setting.",
                )
            elif field_name == "credential_policy":
                group.add_argument(
                    option,
                    dest=dest,
                    choices=("deny", "allow_explicit"),
                    default=None,
                    help="Trusted shell Git credential policy.",
                )
            elif isinstance(field.default, bool):
                group.add_argument(
                    option,
                    dest=dest,
                    action="store_true",
                    default=None,
                    help="Trusted shell Git policy flag.",
                )
                group.add_argument(
                    f"--no-tool-shell-git-{field_name.replace('_', '-')}",
                    dest=dest,
                    action="store_false",
                )
            elif isinstance(field.default, int):
                group.add_argument(
                    option,
                    dest=dest,
                    type=int,
                    default=None,
                    help="Trusted shell Git integer cap.",
                )
            elif isinstance(field.default, float):
                group.add_argument(
                    option,
                    dest=dest,
                    type=float,
                    default=None,
                    help="Trusted shell Git timeout cap.",
                )
            else:
                group.add_argument(
                    option,
                    dest=dest,
                    type=str,
                    default=None,
                    help="Trusted shell Git scalar setting.",
                )

    @staticmethod
    def _add_skills_settings_arguments(
        parser: ArgumentParser,
    ) -> _ArgumentGroup:
        group = parser.add_argument_group("skills tool settings")
        group.add_argument(
            "--tool-skills-source",
            dest="tool_skills_source",
            action="append",
            type=str,
            default=None,
            metavar="LABEL=PATH",
            help="Add a trusted filesystem skills source.",
        )
        group.add_argument(
            "--tool-skills-file",
            dest="tool_skills_file",
            action="append",
            type=str,
            default=None,
            metavar="LABEL=PATH",
            help="Add a trusted skills manifest file source.",
        )
        group.add_argument(
            "--tool-skills-file-no-auto-enable",
            dest="tool_skills_file_no_auto_enable",
            action="store_true",
            default=None,
            help="Do not allow manifest file labels as skill IDs by default.",
        )
        group.add_argument(
            "--tool-skills-source-authority",
            dest="tool_skills_source_authority",
            action="append",
            type=str,
            default=None,
            metavar="LABEL=KIND[:ID]",
            help="Assign a trusted authority to a skills source.",
        )
        group.add_argument(
            "--tool-skills-source-package",
            dest="tool_skills_source_package",
            action="append",
            type=str,
            default=None,
            metavar="LABEL=PATH",
            help="Select a package directory inside a trusted source.",
        )
        group.add_argument(
            "--tool-skills-source-allow-hidden",
            dest="tool_skills_source_allow_hidden",
            action="append",
            type=str,
            default=None,
            metavar="LABEL",
            help="Allow hidden paths inside a trusted skills source.",
        )
        group.add_argument(
            "--tool-skills-authority-kind",
            dest="tool_skills_authority_kind",
            action="append",
            choices=[kind.value for kind in SkillSourceAuthorityKind],
            default=None,
            help="Restrict trusted skills source authority kinds.",
        )
        group.add_argument(
            "--tool-skills-skill",
            dest="tool_skills_skill",
            action="append",
            type=str,
            default=None,
            help="Allow only a specific logical skill ID.",
        )
        group.add_argument(
            "--tool-skills-disable",
            dest="tool_skills_disabled",
            action="store_true",
            default=None,
            help="Disable trusted skills settings.",
        )
        group.add_argument(
            "--tool-skills-bootstrap",
            dest="tool_skills_bootstrap",
            choices=("auto", "off"),
            default=None,
            help="Skills bootstrap prompt mode.",
        )
        group.add_argument(
            "--tool-skills-bootstrap-omit",
            dest="tool_skills_bootstrap_omit",
            action="append",
            choices=(
                "tool_summary",
                "discovery_guidance",
                "read_guidance",
                "check_guidance",
                "behavior_guidance",
            ),
            default=None,
            help="Omit a trusted default skills bootstrap prompt section.",
        )
        group.add_argument(
            "--tool-skills-bootstrap-instruction",
            dest="tool_skills_bootstrap_instruction",
            action="append",
            type=str,
            default=None,
            help="Append a trusted skills bootstrap instruction.",
        )
        group.add_argument(
            "--tool-skills-diagnostics",
            dest="tool_skills_diagnostics",
            choices=("off", "standard", "verbose"),
            default=None,
            help="Skills diagnostic verbosity.",
        )
        group.add_argument(
            "--tool-skills-observability",
            dest="tool_skills_observability",
            choices=("off", "standard", "verbose"),
            default=None,
            help="Skills observability verbosity.",
        )
        group.add_argument(
            "--tool-skills-max-bytes-per-read",
            dest="tool_skills_max_bytes_per_read",
            type=int,
            default=None,
            help="Maximum bytes returned by one skills read.",
        )
        group.add_argument(
            "--tool-skills-max-lines-per-read",
            dest="tool_skills_max_lines_per_read",
            type=int,
            default=None,
            help="Maximum lines returned by one skills read.",
        )
        group.add_argument(
            "--tool-skills-max-skills",
            dest="tool_skills_max_skills",
            type=int,
            default=None,
            help="Maximum indexed skills.",
        )
        group.add_argument(
            "--tool-skills-max-resources-per-skill",
            dest="tool_skills_max_resources_per_skill",
            type=int,
            default=None,
            help="Maximum declared resources per skill.",
        )
        group.add_argument(
            "--tool-skills-max-indexed-bytes",
            dest="tool_skills_max_indexed_bytes",
            type=int,
            default=None,
            help="Maximum indexed skills bytes.",
        )
        group.add_argument(
            "--tool-skills-max-sources",
            dest="tool_skills_max_sources",
            type=int,
            default=None,
            help="Maximum trusted skills sources.",
        )
        group.add_argument(
            "--tool-skills-max-resources-per-source",
            dest="tool_skills_max_resources_per_source",
            type=int,
            default=None,
            help="Maximum resources scanned per skills source.",
        )
        group.add_argument(
            "--tool-skills-max-source-depth",
            dest="tool_skills_max_source_depth",
            type=int,
            default=None,
            help="Maximum directory depth scanned per skills source.",
        )
        group.add_argument(
            "--tool-skills-max-files-per-source",
            dest="tool_skills_max_files_per_source",
            type=int,
            default=None,
            help="Maximum files scanned per skills source.",
        )
        group.add_argument(
            "--tool-skills-max-directory-entries-per-source",
            dest="tool_skills_max_directory_entries_per_source",
            type=int,
            default=None,
            help="Maximum directory entries scanned per skills source.",
        )
        group.add_argument(
            "--tool-skills-max-active-cursors",
            dest="tool_skills_max_active_cursors",
            type=int,
            default=None,
            help="Maximum active skills read cursors.",
        )
        group.add_argument(
            "--tool-skills-max-cursor-age-seconds",
            dest="tool_skills_max_cursor_age_seconds",
            type=int,
            default=None,
            help="Maximum skills read cursor age in seconds.",
        )
        return group

    @staticmethod
    def _add_tool_name_policy_arguments(
        parser: ArgumentParser,
    ) -> _ArgumentGroup:
        group = parser.add_argument_group("tool name policy")
        group.add_argument(
            "--tool-name-policy",
            choices=[mode.value for mode in ToolNamePolicyMode],
            default=None,
            help="Provider-facing tool name policy.",
        )
        group.add_argument(
            "--tool-name-prefix",
            default=None,
            help="Prefix for encoded provider-facing tool names.",
        )
        group.add_argument(
            "--tool-name-replacement",
            default=None,
            help="Replacement text for sanitized provider-facing tool names.",
        )
        group.add_argument(
            "--tool-name-no-collapse-replacement",
            dest="tool_name_collapse_replacement",
            action="store_false",
            default=None,
            help="Do not collapse repeated sanitized-name replacements.",
        )
        group.add_argument(
            "--tool-name-map",
            action="append",
            default=None,
            metavar="CANONICAL=PROVIDER",
            help="Map a canonical tool name to a provider-facing name.",
        )
        return group

    @staticmethod
    def _parse_shell_executable_path(value: str) -> tuple[str, str]:
        """Parse a shell executable mapping CLI value."""
        command, separator, executable = value.partition("=")
        if not separator or not command or not executable:
            raise ArgumentTypeError(
                "expected COMMAND=/absolute/executable/path"
            )
        if not Path(executable).is_absolute():
            raise ArgumentTypeError("shell executable path must be absolute")
        return command, executable

    @staticmethod
    async def _needs_hf_token(args: Namespace) -> bool:
        """Return ``True`` if the command needs hub authentication."""
        command = args.command
        if command == "flow":
            return False
        if command == "task":
            engine = await CLI._task_agent_engine_uri(args)
            if not engine:
                return False
            engine_uri = ModelManager.parse_uri(engine)
            if engine_uri.is_local and is_ds4_backend_selected(
                args, engine_uri
            ):
                return False
            return bool(engine_uri.is_local)
        if command == "model" and (args.model_command or "display") == "run":
            assert isinstance(args.model, str)
            engine_uri = ModelManager.parse_uri(args.model)
            if engine_uri.is_local and is_ds4_backend_selected(
                args, engine_uri
            ):
                return False
            return bool(engine_uri.is_local)
        if command == "agent" and (
            (args.agent_command or "run") in {"run", "serve", "proxy"}
        ):
            engine = getattr(args, "engine_uri", None)
            if engine:
                assert isinstance(engine, str)
                engine_uri = ModelManager.parse_uri(engine)
                if engine_uri.is_local and is_ds4_backend_selected(
                    args, engine_uri
                ):
                    return False
                return bool(engine_uri.is_local)
            specs = getattr(args, "specifications_file", None)
            if specs:
                config = toml_loads(await read_text(specs))
                engine_uri_str = config.get("engine", {}).get("uri")
                if engine_uri_str:
                    assert isinstance(engine_uri_str, str)
                    engine_uri = ModelManager.parse_uri(engine_uri_str)
                    if engine_uri.is_local and is_ds4_backend_selected(
                        args, engine_uri
                    ):
                        return False
                    return bool(engine_uri.is_local)
        return True

    @staticmethod
    async def _task_agent_engine_uri(args: Namespace) -> str | None:
        task_command = args.task_command or "validate"
        if task_command not in {"enqueue", "run"}:
            return None
        definition = getattr(args, "definition", None)
        if not isinstance(definition, str):
            return None
        try:
            task_config = toml_loads(await read_text(definition))
            execution = task_config.get("execution")
            if not isinstance(execution, dict):
                return None
            if execution.get("type") != "agent":
                return None
            ref = execution.get("ref")
            if not isinstance(ref, str):
                return None
            agent_path = Path(definition).parent / ref
            agent_config = toml_loads(await read_text(agent_path))
            engine = agent_config.get("engine")
            if not isinstance(engine, dict):
                return None
            uri = engine.get("uri")
            return uri if isinstance(uri, str) else None
        except (OSError, TOMLDecodeError):
            return None

    @staticmethod
    def _can_use_anonymous_hub(args: Namespace) -> bool:
        command = args.command
        if command == "task":
            return (args.task_command or "validate") != "worker"
        if command != "model" or (args.model_command or "display") != "run":
            return False
        assert isinstance(args.model, str)
        engine_uri = ModelManager.parse_uri(args.model)
        return not bool(engine_uri.is_local) or is_ds4_backend_selected(
            args, engine_uri
        )

    async def __call__(self) -> None:
        argv, chat_opts = self._extract_chat_settings(sys.argv[1:])
        args = self._parser.parse_args(argv)

        if args.parallel and not args.quiet:
            args.quiet = True

        if args.parallel and "LOCAL_RANK" not in environ:
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc-per-node",
                str(args.parallel_count),
                "-m",
                "avalan.cli",
            ] + argv
            run(cmd, check=True)
            return

        if args.parallel and "LOCAL_RANK" in environ:
            rank = int(environ["LOCAL_RANK"])
            if args.device.startswith("cuda") and ":" not in args.device:
                args.device = f"cuda:{rank}"
                set_device(rank)

        if args.version:
            print(f"{self._name} {self._version}")
            return

        for key, value in chat_opts.items():
            setattr(args, f"run_chat_{key}", value)

        translator = CLI._get_translator(self._name, args.locales, args.locale)

        assert self._logger is not None and isinstance(self._logger, Logger)

        theme_name = getattr(args, "theme", DEFAULT_THEME_NAME)
        theme = create_theme(
            theme_name,
            translator.gettext,
            translator.ngettext,
        )
        _ = theme._
        rich_theme_styles = {
            str(data_key): style
            for data_key, style in theme.get_styles().items()
        }
        console = Console(
            theme=RichTheme(styles=rich_theme_styles),
            record=args.record and not args.quiet,
        )
        self._abort_console = console
        self._abort_quiet = args.quiet
        self._abort_theme = theme

        if args.help_full:
            return self._help(console, self._parser)

        access_token = args.hf_token
        requires_token = await self._needs_hf_token(args)

        if requires_token:
            if not access_token:
                prompt_stream: TextIO | None = None
                if has_input(console):
                    try:
                        prompt_stream = open(args.tty)
                    except OSError:
                        pass
                access_token = Prompt.ask(
                    theme.ask_access_token(), stream=prompt_stream
                )
            assert access_token
        else:
            access_token = access_token or "anonymous"

        hub: HubClient
        if (
            requires_token
            or bool(args.hf_token)
            or args.login
            or not self._can_use_anonymous_hub(args)
        ):
            hub_class = _huggingface_hub_class()
            hub = hub_class(access_token, args.cache_dir, self._logger)
        else:
            hub = _AnonymousHub(args.cache_dir)

        try:
            with _direct_keyboard_interrupts():
                try:
                    await self._main(args, theme, console, hub)
                except (
                    CancelledError,
                    KeyboardInterrupt,
                    CommandAbortException,
                ):
                    self._print_bye(console, theme, quiet=args.quiet)
                    raise
        finally:
            if args.parallel and "LOCAL_RANK" in environ:
                try:
                    destroy_process_group()
                except AssertionError:
                    # Process group might be dead already
                    pass

    def _print_bye(
        self,
        console: Console | None = None,
        theme: Theme | None = None,
        *,
        quiet: bool | None = None,
    ) -> None:
        if self._abort_printed:
            return

        is_quiet = self._abort_quiet if quiet is None else quiet
        if is_quiet:
            return

        self._abort_printed = True
        bye_console = console or self._abort_console or Console()
        bye_theme = theme or self._abort_theme
        bye_console.print(
            bye_theme.bye() if bye_theme else ":vulcan_salute: bye :)"
        )

    def _help(
        self,
        console: Console,
        parser: ArgumentParser,
        path: list[str] | None = None,
    ) -> None:
        """Recursively output help information for ``parser``."""
        if path is None:
            path = []

        prog = parser.prog
        is_root_command = not path
        console.print(
            ("#" if is_root_command else "#" * (len(path) + 1)) + f" {prog}"
        )
        console.print("")
        console.print("```")
        console.print(parser.format_help().strip())
        console.print("```")
        console.print("")
        for action in parser._actions:
            if isinstance(action, _SubParsersAction):
                for name, subparser in action.choices.items():
                    self._help(console, subparser, path + [name])

    async def _main(
        self,
        args: Namespace,
        theme: Theme,
        console: Console,
        hub: HubClient,
        suggest_login: bool = False,
    ) -> None:
        user: User | None = None
        _ = theme._

        verbosity = args.verbose or 0
        log_level = (
            DEBUG if verbosity >= 2 else INFO if verbosity >= 1 else WARNING
        )
        previous_log_level = self._logger.getEffectiveLevel()

        self._logger.setLevel(log_level)

        if find_spec("sentence_transformers"):
            logger_replace(self._logger, ["sentence_transformers"])

        if find_spec("httpx"):
            logger_replace(self._logger, ["httpx"])

        filterwarnings(
            "ignore",
            message=r".*`do_sample` is set to `False`. "
            r"However, `temperature` is set.*",
        )

        class _SilencingFilter(Filter):
            def filter(self, record: LogRecord) -> bool:
                message = record.getMessage()
                return (
                    "Some weights of the model checkpoint" not in message
                    or not "BertForTokenClassification"
                ) and "wav2vec2.masked_spec_embed" not in message

        hf_logger = hf_logging.get_logger("transformers.modeling_utils")
        hf_logger.addFilter(_SilencingFilter())

        filterwarnings(
            "ignore",
            message=r".*Some weights of Wav2Vec2ForCTC were not initialized.*",
        )

        suggest_login = suggest_login and not has_input(console)
        if args.login or (
            suggest_login
            and Confirm.ask(theme.ask_login_to_hub(), default=False)
        ):
            with console.status(
                theme.logging_in(hub.domain),
                spinner=(theme.get_spinner("connecting") or "dots"),
                refresh_per_second=self._REFRESH_RATE,
            ):
                hub.login()
                user = hub.user()

        if not args.quiet and not _task_run_json_stdout(args):
            console.print(
                theme.welcome(
                    self._site.geturl(),
                    self._name,
                    str(self._version),
                    self._license,
                    user,
                )
            )

        match args.command:
            case "agent":
                subcommand = args.agent_command or "run"
                match subcommand:
                    case "message":
                        innercommand = args.agent_message_command or "search"
                        match innercommand:
                            case "search":
                                await agent_message_search(
                                    args,
                                    console,
                                    theme,
                                    hub,
                                    self._logger,
                                    refresh_per_second=self._REFRESH_RATE,
                                )
                    case "run":
                        await agent_run(
                            args,
                            console,
                            theme,
                            hub,
                            self._logger,
                            refresh_per_second=self._REFRESH_RATE,
                        )
                    case "serve":
                        await agent_serve(
                            args,
                            hub,
                            self._logger,
                            self._name,
                            str(self._version),
                        )
                    case "proxy":
                        await agent_proxy(
                            args,
                            hub,
                            self._logger,
                            self._name,
                            str(self._version),
                        )
                    case "init":
                        await agent_init(args, console, theme)
            case "cache":
                subcommand = args.cache_command or "list"
                match subcommand:
                    case "delete":
                        cache_delete(args, console, theme, hub)
                    case "download":
                        cache_download(args, console, theme, hub)
                    case "list":
                        cache_list(args, console, theme, hub)
            case "memory":
                subcommand = args.memory_command or "embeddings"
                match subcommand:
                    case "document":
                        innercommand = args.memory_document_command or "index"
                        match innercommand:
                            case "index":
                                await memory_document_index(
                                    args, console, theme, hub, self._logger
                                )
                    case "search":
                        await memory_search(
                            args, console, theme, hub, self._logger
                        )
                    case "embeddings":
                        await memory_embeddings(
                            args, console, theme, hub, self._logger
                        )
            case "model":
                subcommand = args.model_command or "display"
                match subcommand:
                    case "display":
                        model_display(args, console, theme, hub, self._logger)
                    case "install":
                        model_install(args, console, theme, hub)
                    case "run":
                        await model_run(
                            args,
                            console,
                            theme,
                            hub,
                            self._REFRESH_RATE,
                            self._logger,
                        )
                    case "search":
                        await model_search(
                            args, console, theme, hub, self._REFRESH_RATE
                        )
                    case "uninstall":
                        model_uninstall(args, console, theme, hub)
            case "deploy":
                subcommand = args.deploy_command or "run"
                match subcommand:
                    case "run":
                        await deploy_run(args, self._logger)
            case "flow":
                subcommand = args.flow_command or "run"
                match subcommand:
                    case "cancel":
                        if not flow_cancel(args, console, theme):
                            raise SystemExit(1)
                    case "compile":
                        if not flow_compile(args, console, theme):
                            raise SystemExit(1)
                    case "graph":
                        if not flow_graph(args, console, theme):
                            raise SystemExit(1)
                    case "inspect":
                        if not flow_inspect(args, console, theme):
                            raise SystemExit(1)
                    case "mermaid":
                        if not flow_mermaid(args, console, theme):
                            raise SystemExit(1)
                    case "resume":
                        if not flow_resume(args, console, theme):
                            raise SystemExit(1)
                    case "run":
                        if not flow_run(
                            args,
                            console,
                            theme,
                            hub,
                            self._logger,
                        ):
                            raise SystemExit(1)
                    case "trace":
                        if not flow_trace(args, console, theme):
                            raise SystemExit(1)
                    case "validate":
                        if not flow_validate(args, console, theme):
                            raise SystemExit(1)
            case "task":
                subcommand = args.task_command or "validate"
                match subcommand:
                    case "artifacts":
                        task_artifacts(args, console, theme)
                    case "enqueue":
                        task_enqueue(args, console, theme, hub, self._logger)
                    case "events":
                        task_events(args, console, theme)
                    case "inspect":
                        task_inspect(args, console, theme)
                    case "usage":
                        task_usage(args, console, theme)
                    case "output":
                        task_output(args, console, theme)
                    case "retention-sweep":
                        task_retention_sweep(args, console, theme)
                    case "validate":
                        if not task_validate(args, console, theme):
                            raise SystemExit(1)
                    case "pgsql":
                        pgsql_subcommand = args.task_pgsql_command or "status"
                        match pgsql_subcommand:
                            case "check":
                                task_pgsql_check(args, console, theme)
                            case "diagnose":
                                task_pgsql_diagnose(args, console, theme)
                            case "migrate":
                                task_pgsql_migrate(args, console, theme)
                            case "stamp":
                                task_pgsql_stamp(args, console, theme)
                            case "status":
                                task_pgsql_status(args, console, theme)
                    case "run":
                        if not task_run(
                            args,
                            console,
                            theme,
                            hub,
                            self._logger,
                        ):
                            raise SystemExit(1)
                    case "worker":
                        task_worker(args, console, theme, hub, self._logger)

            case "tokenizer":
                await tokenize(args, console, theme, hub, self._logger)

        self._logger.setLevel(previous_log_level)


def main() -> None:
    """Entry point for the ``avalan`` CLI."""
    if "--version" in sys.argv[1:]:
        print(f"{name()} {version()}")
        return

    environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    basicConfig(
        level=INFO,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger = getLogger(name())

    cli = CLI(logger)
    try:
        run_in_loop(cli())
    except (CancelledError, KeyboardInterrupt, CommandAbortException):
        cli._print_bye()


if __name__ == "__main__":
    main()  # pragma: no cover
