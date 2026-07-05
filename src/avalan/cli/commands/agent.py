from ...agent.loader import OrchestratorLoader
from ...agent.orchestrator import Orchestrator
from ...agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from ...cli import get_input, has_input
from ...cli.commands.model import token_generation
from ...cli.display import cli_stream_display_config
from ...cli.stream_coordinator import CliStreamCoordinator
from ...cli.theme import Theme
from ...container import (
    ContainerAsyncBackend,
    ContainerProfileSelection,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerToolRuntimeSettings,
    ContainerTrustLevel,
    container_selection_from_mapping,
    trusted_container_runtime_from_mapping,
)
from ...entities import (
    Backend,
    EngineMessageScored,
    GenerationCacheStrategy,
    Message,
    Model,
    OrchestratorSettings,
    PermanentMemoryStoreSettings,
    ToolCall,
    ToolCallRecoveryFormat,
    ToolFormat,
    ToolNamePolicyMode,
    ToolNamePolicySettings,
)
from ...event import EventStats, EventType
from ...event.manager import EventManagerMode
from ...isolation import (
    IsolationMode,
    IsolationProfileSelection,
    IsolationSettingsSource,
    IsolationSettingsSurface,
    IsolationToolRuntimeSettings,
    SandboxProfileSelection,
    trusted_isolation_runtime_from_mapping,
    trusted_isolation_source,
)
from ...model.hubs.huggingface import HuggingfaceHub
from ...model.input import input_files
from ...model.nlp.text.generation import TextGenerationModel
from ...model.nlp.text.vendor import TextGenerationVendorModel
from ...model.response.text import TextGenerationResponse
from ...sandbox import (
    BubblewrapSandboxBackend,
    SandboxAsyncBackend,
    SeatbeltSandboxBackend,
)
from ...server import agents_server
from ...server.entities import (
    ServerOutputRedactionChannel,
    ServerOutputRedactionProtocol,
    ServerOutputRedactionRule,
    ServerOutputRedactionSettings,
)
from ...skill import (
    BundledSkillSourceAuthority,
    PluginProvidedSkillSourceAuthority,
    PreinstalledRemoteSkillSourceAuthority,
    SkillBootstrapPromptSettings,
    SkillCursorLimits,
    SkillIndexLimits,
    SkillObservabilitySettings,
    SkillPrivacySettings,
    SkillReadLimits,
    SkillSourceAuthority,
    SkillSourceAuthorityKind,
    SkillSourceConfig,
    SkillSourceLimits,
    TrustedSkillSettings,
    UserLocalSkillSourceAuthority,
    WorkspaceSkillSourceAuthority,
)
from ...tool.browser import BrowserToolSettings
from ...tool.context import ToolSettingsContext
from ...tool.database.settings import DatabaseToolSettings
from ...tool.graph_settings import GraphToolSettings
from ...tool.shell import ShellGitToolSettings, ShellToolSettings
from ...tool_cycles import MaximumToolCycles

from argparse import Namespace
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import fields
from importlib import import_module
from json import dumps as json_dumps
from logging import Logger
from os.path import dirname, getmtime, join
from typing import Any, cast, overload
from uuid import UUID, uuid4

from jinja2 import Environment, FileSystemLoader
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax


class _Unset:
    pass


_UNSET = _Unset()
_APPLE_CONTAINER_BACKEND = "apple-container"
_APPLE_CONTAINER_BACKEND_MODULES = (
    "avalan.container",
    "avalan.container.apple",
    "avalan.container.apple_container",
)
_DOCKER_CONTAINER_BACKEND = "docker"
_DOCKER_CONTAINER_BACKEND_MODULES = (
    "avalan.container",
    "avalan.container.docker",
)
_SUPPORTED_CONTAINER_BACKENDS = frozenset({"apple-container", "docker"})
_BUBBLEWRAP_SANDBOX_BACKEND = "bubblewrap"
_SEATBELT_SANDBOX_BACKEND = "seatbelt"
_SUPPORTED_SANDBOX_BACKENDS = frozenset({"bubblewrap", "seatbelt"})
_SKILL_BOOTSTRAP_PROMPT_OMIT_FIELDS = {
    "tool_summary": "include_tool_summary",
    "discovery_guidance": "include_discovery_guidance",
    "read_guidance": "include_read_guidance",
    "check_guidance": "include_check_guidance",
    "behavior_guidance": "include_behavior_guidance",
}
_SHELL_GIT_CLI_FIELD_NAMES = frozenset(
    (
        *ShellGitToolSettings.CLI_SCALAR_FIELDS,
        *ShellGitToolSettings.CLI_SEQUENCE_FIELDS,
    )
)


def _parse_permanent_memory_items(
    items: Iterable[str],
) -> dict[str, PermanentMemoryStoreSettings]:
    stores: dict[str, PermanentMemoryStoreSettings] = {}
    for item in items:
        namespace, value = item.split("@", 1)
        namespace = namespace.strip()
        assert namespace, "Permanent memory namespace must be provided"
        stores[namespace] = OrchestratorLoader.parse_permanent_store_value(
            value
        )
    return stores


def get_orchestrator_settings(
    args: Namespace,
    *,
    agent_id: UUID,
    name: str | None = None,
    role: str | None = None,
    task: str | None = None,
    instructions: str | None = None,
    goal_instructions: str | None = None,
    system: str | None = None,
    developer: str | None = None,
    user: str | None = None,
    user_template: str | None = None,
    engine_uri: str | None = None,
    memory_recent: bool | None = None,
    memory_permanent_message: str | None = None,
    memory_permanent: list[str] | None = None,
    maximum_tool_cycles: MaximumToolCycles | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    tools: list[str] | None | _Unset = _UNSET,
    tool_choice: str | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool | None = None,
    cache_strategy: GenerationCacheStrategy | None = None,
) -> OrchestratorSettings:
    """Create ``OrchestratorSettings`` from CLI arguments."""
    assert not (
        (user or getattr(args, "user", None))
        and (user_template or getattr(args, "user_template", None))
    )
    memory_recent = (
        memory_recent
        if memory_recent is not None
        else (
            args.memory_recent
            if args.memory_recent is not None
            else not getattr(args, "no_session", False)
        )
    )
    engine_uri = engine_uri or args.engine_uri
    call_tokens = (
        max_new_tokens
        if max_new_tokens is not None
        else args.run_max_new_tokens
    )
    call_maximum_tool_cycles = (
        maximum_tool_cycles
        if maximum_tool_cycles is not None
        else getattr(args, "run_maximum_tool_cycles", None)
    )
    call_tool_choice = (
        tool_choice
        if tool_choice is not None
        else getattr(args, "tool_choice", None)
    )

    chat_settings = {
        k[len("run_chat_") :]: v
        for k, v in vars(args).items()
        if k.startswith("run_chat_") and v is not None
    }
    reasoning_settings = {
        k[len("run_reasoning_") :]: v
        for k, v in vars(args).items()
        if k.startswith("run_reasoning_") and v is not None
    }
    call_options = {
        "max_new_tokens": call_tokens,
        "skip_special_tokens": args.run_skip_special_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        **({"chat_settings": chat_settings} if chat_settings else {}),
        **({"reasoning": reasoning_settings} if reasoning_settings else {}),
    }
    if use_cache is not None:
        call_options["use_cache"] = use_cache
    if cache_strategy is not None:
        call_options["cache_strategy"] = cache_strategy
    if call_maximum_tool_cycles is not None:
        call_options["maximum_tool_cycles"] = call_maximum_tool_cycles
    if call_tool_choice is not None:
        call_options["tool_choice"] = call_tool_choice
    engine_config: dict[str, Any] = {
        "backend": getattr(args, "backend", Backend.TRANSFORMERS.value)
    }
    engine_base_url = getattr(args, "engine_base_url", None)
    if engine_base_url is not None:
        engine_config["base_url"] = engine_base_url

    return OrchestratorSettings(
        agent_id=agent_id,
        orchestrator_type=None,
        agent_config={
            k: v
            for k, v in {
                "name": name if name is not None else args.name,
                "role": role if role is not None else args.role,
                "task": task if task is not None else args.task,
                "instructions": (
                    instructions
                    if instructions is not None
                    else getattr(args, "instructions", None)
                ),
                "goal_instructions": (
                    goal_instructions
                    if goal_instructions is not None
                    else getattr(args, "goal_instructions", None)
                ),
                "system": (
                    system
                    if system is not None
                    else getattr(args, "system", None)
                ),
                "developer": (
                    developer
                    if developer is not None
                    else getattr(args, "developer", None)
                ),
                "user": (
                    user if user is not None else getattr(args, "user", None)
                ),
                "user_template": (
                    user_template
                    if user_template is not None
                    else getattr(args, "user_template", None)
                ),
            }.items()
            if v is not None
        },
        uri=engine_uri,
        engine_config=engine_config,
        call_options=call_options,
        template_vars=None,
        memory_permanent_message=(
            memory_permanent_message
            if memory_permanent_message is not None
            else args.memory_permanent_message
        ),
        permanent_memory=(
            _parse_permanent_memory_items(memory_permanent)
            if memory_permanent is not None
            else (
                _parse_permanent_memory_items(args.memory_permanent)
                if args.memory_permanent
                else None
            )
        ),
        memory_recent=memory_recent,
        sentence_model_id=(
            args.memory_engine_model_id
            or OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID
        ),
        sentence_model_engine_config=None,
        sentence_model_max_tokens=args.memory_engine_max_tokens,
        sentence_model_overlap_size=args.memory_engine_overlap,
        sentence_model_window_size=args.memory_engine_window,
        json_config=None,
        tools=(
            tools
            if not isinstance(tools, _Unset)
            else (args.tool or []) + (getattr(args, "tools", None) or [])
        ),
        log_events=True,
    )


def _tool_settings_from_mapping(
    mapping: Mapping[str, object] | Namespace,
    *,
    prefix: str | None = None,
    settings_cls: (
        type[BrowserToolSettings]
        | type[DatabaseToolSettings]
        | type[GraphToolSettings]
        | type[ShellToolSettings]
    ),
    open_files: bool = True,
) -> (
    BrowserToolSettings
    | DatabaseToolSettings
    | GraphToolSettings
    | ShellToolSettings
    | None
):
    """Return tool settings from a mapping using dataclass ``settings_cls``."""
    values: dict[str, object] = {}
    for field in fields(settings_cls):
        key = f"tool_{prefix}_{field.name}" if prefix else field.name
        if isinstance(mapping, Namespace):
            if hasattr(mapping, key):
                value = getattr(mapping, key)
            else:
                continue
        else:
            if key in mapping:
                value = mapping[key]
            elif prefix and field.name in mapping:
                value = mapping[field.name]
            else:
                continue

        if value is not None:
            if (
                field.name == "debug_source"
                and open_files
                and isinstance(value, str)
            ):
                value = open(value)
            if settings_cls is DatabaseToolSettings:
                value = _coerce_database_tool_setting_value(
                    field.name,
                    value,
                    from_cli=isinstance(mapping, Namespace),
                )
            if settings_cls is ShellToolSettings:
                value = _coerce_shell_tool_setting_value(field.name, value)
            values[field.name] = value

    if settings_cls is ShellToolSettings:
        git_settings = _shell_git_settings_from_mapping(mapping)
        if git_settings is not None:
            values["git"] = git_settings

    if not values:
        return None

    settings = cast(
        BrowserToolSettings
        | DatabaseToolSettings
        | GraphToolSettings
        | ShellToolSettings,
        cast(Any, settings_cls)(**values),
    )
    return settings


def _tool_settings_explicit_fields_from_mapping(
    mapping: Mapping[str, object] | Namespace,
    *,
    prefix: str | None = None,
    settings_cls: (
        type[BrowserToolSettings]
        | type[DatabaseToolSettings]
        | type[GraphToolSettings]
        | type[ShellToolSettings]
    ),
) -> frozenset[str]:
    """Return dataclass fields explicitly present in tool settings input."""
    explicit_fields: set[str] = set()
    for field in fields(settings_cls):
        key = f"tool_{prefix}_{field.name}" if prefix else field.name
        if isinstance(mapping, Namespace):
            if hasattr(mapping, key) and getattr(mapping, key) is not None:
                explicit_fields.add(field.name)
        elif key in mapping and mapping[key] is not None:
            explicit_fields.add(field.name)
        elif (
            prefix
            and field.name in mapping
            and mapping[field.name] is not None
        ):
            explicit_fields.add(field.name)
    if settings_cls is ShellToolSettings:
        explicit_fields.update(
            _shell_git_explicit_fields_from_mapping(mapping)
        )
    return frozenset(explicit_fields)


def _coerce_database_tool_setting_value(
    field_name: str, value: object, *, from_cli: bool
) -> object:
    if field_name != "allowed_commands":
        return value
    if from_cli and isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return list(value)
    return value


def _coerce_shell_tool_setting_value(field_name: str, value: object) -> object:
    if field_name == "executable_paths":
        return _coerce_shell_executable_paths(value)
    return value


def _shell_git_settings_from_mapping(
    mapping: Mapping[str, object] | Namespace,
) -> ShellGitToolSettings | None:
    values: dict[str, object] = {}
    for field_name in _SHELL_GIT_CLI_FIELD_NAMES:
        key = f"tool_shell_git_{field_name}"
        if isinstance(mapping, Namespace):
            if not hasattr(mapping, key):
                continue
            value = getattr(mapping, key)
        else:
            if key not in mapping:
                continue
            value = mapping[key]
        if value is not None:
            values[field_name] = value
    if not values:
        return None
    _complete_partial_shell_git_timeout_values(values)
    return cast(
        ShellGitToolSettings, cast(Any, ShellGitToolSettings)(**values)
    )


def _complete_partial_shell_git_timeout_values(
    values: dict[str, object],
) -> None:
    default_key = "default_timeout_seconds"
    max_key = "max_timeout_seconds"
    if default_key in values and max_key in values:
        return

    defaults = ShellGitToolSettings()
    if max_key in values:
        max_timeout = values[max_key]
        if (
            isinstance(max_timeout, int | float)
            and not isinstance(max_timeout, bool)
            and max_timeout < defaults.default_timeout_seconds
        ):
            values[default_key] = max_timeout
    elif default_key in values:
        default_timeout = values[default_key]
        if (
            isinstance(default_timeout, int | float)
            and not isinstance(default_timeout, bool)
            and default_timeout > defaults.max_timeout_seconds
        ):
            values[max_key] = default_timeout


def _shell_git_explicit_fields_from_mapping(
    mapping: Mapping[str, object] | Namespace,
) -> set[str]:
    explicit_fields: set[str] = set()
    for field_name in _SHELL_GIT_CLI_FIELD_NAMES:
        key = f"tool_shell_git_{field_name}"
        if isinstance(mapping, Namespace):
            if hasattr(mapping, key) and getattr(mapping, key) is not None:
                explicit_fields.add(f"git.{field_name}")
        elif key in mapping and mapping[key] is not None:
            explicit_fields.add(f"git.{field_name}")
    return explicit_fields


def _coerce_shell_executable_paths(value: object) -> object:
    if isinstance(value, Mapping):
        return value
    if not _is_tuple_pair_sequence(value):
        return value

    executable_paths: dict[str, str] = {}
    for command, executable in cast(Sequence[tuple[str, str]], value):
        executable_paths[command] = executable
    return executable_paths


def _is_tuple_pair_sequence(
    value: object,
) -> bool:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return False
    return all(
        isinstance(item, tuple)
        and len(item) == 2
        and all(isinstance(part, str) for part in item)
        for item in value
    )


@overload
def get_tool_settings(
    args: Namespace,
    *,
    prefix: str,
    settings_cls: type[BrowserToolSettings],
    open_files: bool = True,
) -> BrowserToolSettings | None: ...


@overload
def get_tool_settings(
    args: Namespace,
    *,
    prefix: str,
    settings_cls: type[DatabaseToolSettings],
    open_files: bool = True,
) -> DatabaseToolSettings | None: ...


@overload
def get_tool_settings(
    args: Namespace,
    *,
    prefix: str,
    settings_cls: type[GraphToolSettings],
    open_files: bool = True,
) -> GraphToolSettings | None: ...


@overload
def get_tool_settings(
    args: Namespace,
    *,
    prefix: str,
    settings_cls: type[ShellToolSettings],
    open_files: bool = True,
) -> ShellToolSettings | None: ...


def get_tool_settings(
    args: Namespace,
    *,
    prefix: str,
    settings_cls: (
        type[BrowserToolSettings]
        | type[DatabaseToolSettings]
        | type[GraphToolSettings]
        | type[ShellToolSettings]
    ),
    open_files: bool = True,
) -> (
    BrowserToolSettings
    | DatabaseToolSettings
    | GraphToolSettings
    | ShellToolSettings
    | None
):
    return _tool_settings_from_mapping(
        args, prefix=prefix, settings_cls=settings_cls, open_files=open_files
    )


def _shell_tool_template_settings(
    settings: ShellToolSettings | None,
) -> (
    dict[
        str,
        bool | int | float | str | tuple[str, ...] | dict[str, object],
    ]
    | None
):
    if settings is None:
        return None

    default_settings = ShellToolSettings()
    rendered: dict[
        str,
        bool | int | float | str | tuple[str, ...] | dict[str, object],
    ] = {}
    for field in fields(ShellToolSettings):
        name = field.name
        if name == "execution_mode":
            continue
        value = getattr(settings, name)
        if value == getattr(default_settings, name):
            continue
        if name == "git":
            assert isinstance(value, ShellGitToolSettings)
            default_git_settings = default_settings.git
            assert isinstance(default_git_settings, ShellGitToolSettings)
            git_settings = _shell_git_tool_template_settings(
                value,
                default_git_settings,
            )
            if git_settings:
                rendered[name] = git_settings
            continue
        if isinstance(value, bool | int | float | str):
            rendered[name] = value
        elif _is_simple_string_mapping(value):
            rendered[name] = dict(value)
        elif _is_simple_string_sequence(value):
            rendered[name] = tuple(value)
    return rendered


def _shell_git_tool_template_settings(
    settings: ShellGitToolSettings,
    default_settings: ShellGitToolSettings,
) -> dict[str, object]:
    rendered: dict[str, object] = {}
    for field in fields(ShellGitToolSettings):
        name = field.name
        value = getattr(settings, name)
        if value == getattr(default_settings, name):
            continue
        if isinstance(value, bool | int | float | str):
            rendered[name] = value
        elif _is_simple_string_sequence(value):
            rendered[name] = tuple(value)
    return rendered


def _skills_tool_template_settings(
    settings: TrustedSkillSettings | None,
) -> dict[str, object] | None:
    if settings is None:
        return None

    default_settings = TrustedSkillSettings()
    rendered: dict[str, object] = {}
    if settings.enabled != default_settings.enabled:
        rendered["enabled"] = settings.enabled
    if settings.bootstrap_enabled != default_settings.bootstrap_enabled:
        rendered["bootstrap"] = "auto" if settings.bootstrap_enabled else "off"
    if settings.manifest_auto_enable != (
        default_settings.manifest_auto_enable
    ):
        rendered["manifest_auto_enable"] = settings.manifest_auto_enable
    if settings.authority_kinds != default_settings.authority_kinds:
        rendered["authority_kinds"] = tuple(
            authority_kind.value for authority_kind in settings.authority_kinds
        )
    manifest_sources = {
        source.label: _skills_manifest_source_template_value(source)
        for source in settings.sources
        if source.manifest_path is not None
    }
    if manifest_sources:
        rendered["files"] = manifest_sources
    if settings.sources:
        rendered["source_labels"] = tuple(
            source.label for source in settings.sources
        )
    if settings.allowed_skill_ids and settings.allowed_skill_ids_explicit:
        rendered["skill_ids"] = settings.allowed_skill_ids

    for name, value, default_value in (
        ("read_limits", settings.read_limits, default_settings.read_limits),
        ("index_limits", settings.index_limits, default_settings.index_limits),
        (
            "source_limits",
            settings.source_limits,
            default_settings.source_limits,
        ),
        (
            "cursor_limits",
            settings.cursor_limits,
            default_settings.cursor_limits,
        ),
        ("privacy", settings.privacy, default_settings.privacy),
        (
            "observability",
            settings.observability,
            default_settings.observability,
        ),
    ):
        nested = _skills_dataclass_template_settings(value, default_value)
        if nested:
            rendered[name] = nested

    return rendered or None


def _skills_manifest_source_template_value(
    source: SkillSourceConfig,
) -> str | dict[str, object]:
    assert isinstance(source, SkillSourceConfig)
    assert source.manifest_path is not None
    path = str(source.manifest_path)
    authority = _skills_source_authority_template_value(source.authority)
    default_authority = _skills_source_authority_template_value(
        WorkspaceSkillSourceAuthority(),
    )
    if authority == default_authority and not source.allow_hidden_paths:
        return path
    value: dict[str, object] = {"path": path}
    if authority != default_authority:
        value["authority"] = authority
    if source.allow_hidden_paths:
        value["allow_hidden"] = True
    return value


def _skills_source_authority_template_value(
    authority: SkillSourceAuthority,
) -> str:
    assert isinstance(authority, SkillSourceAuthority)
    if isinstance(authority, BundledSkillSourceAuthority):
        return f"bundled:{authority.bundle_id}"
    if isinstance(authority, WorkspaceSkillSourceAuthority):
        return f"workspace:{authority.workspace_id}"
    if isinstance(authority, UserLocalSkillSourceAuthority):
        return f"user_local:{authority.profile_id}"
    if isinstance(authority, PluginProvidedSkillSourceAuthority):
        return f"plugin_provided:{authority.plugin_id}"
    if isinstance(authority, PreinstalledRemoteSkillSourceAuthority):
        return f"preinstalled_remote:{authority.registry_id}"
    raise AssertionError("unsupported skills source authority")


def _skills_dataclass_template_settings(
    settings: Any,
    default_settings: Any,
) -> dict[str, object]:
    rendered: dict[str, object] = {}
    for field in fields(settings):
        name = field.name
        value = getattr(settings, name)
        if value != getattr(default_settings, name):
            rendered[name] = value
    return rendered


def _toml_template_value(value: object) -> str:
    """Return a TOML literal for agent blueprint templates."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json_dumps(value, ensure_ascii=False)
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, Mapping):
        items = ", ".join(
            f"{key} = {_toml_template_value(item)}"
            for key, item in sorted(value.items())
        )
        return "{ " + items + " }"
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return (
            "[" + ", ".join(_toml_template_value(item) for item in value) + "]"
        )
    raise AssertionError("unsupported TOML template value")


def _container_tool_template_settings(
    args: Namespace,
) -> dict[str, object] | None:
    backend = getattr(args, "tool_container_backend", None)
    if not _has_container_policy_args(args):
        return None
    _require_shell_backend_for_policy(
        args,
        None,
        "container",
        "tool.container",
    )
    _agent_container_config_from_args(args)
    profile_name = getattr(args, "tool_container_profile", None) or (
        getattr(args, "tool_shell_container_profile", None)
        or "workspace-readonly"
    )
    profile: dict[str, object] = {}
    for source_key, target_key in (
        ("tool_container_image", "image"),
        ("tool_container_workspace_root", "workspace_root"),
        ("tool_container_pull_policy", "pull_policy"),
        ("tool_container_platform", "platform"),
        ("tool_container_network_mode", "network"),
        ("tool_container_review_mode", "review_mode"),
    ):
        value = getattr(args, source_key, None)
        if value is not None and not _is_default_container_template_value(
            target_key,
            value,
        ):
            profile[target_key] = value
    resources = _agent_container_resources_from_args(args)
    if resources:
        profile["resources"] = resources
    rendered: dict[str, object] = {
        "profiles": {profile_name: profile},
        "default_profile": profile_name,
    }
    if backend is not None:
        rendered["backend"] = backend
    return rendered


def _shell_container_template_settings(
    args: Namespace,
) -> dict[str, object] | None:
    rendered: dict[str, object] = {}
    profile = getattr(args, "tool_shell_container_profile", None)
    if profile is not None:
        rendered["profile"] = profile
    if getattr(args, "tool_shell_container_required", None):
        rendered["required"] = True
    if not rendered:
        return None
    assert (
        _shell_execution_mode_from_args(args) == "container"
    ), "tool.shell.container requires tool.shell backend container"
    container_config = _agent_container_config_from_args(args)
    assert container_config is not None, "container backend is required"
    return rendered


def _sandbox_tool_template_settings(
    args: Namespace,
) -> dict[str, object] | None:
    if not _has_sandbox_policy_args(args):
        return None
    _require_shell_backend_for_policy(args, None, "sandbox", "tool.sandbox")
    sandbox_config = _agent_sandbox_config_from_args(args)
    assert sandbox_config is not None
    _validate_sandbox_template_config(args, sandbox_config)
    rendered: dict[str, object] = {
        "backend": sandbox_config["backend"],
        "profiles": {
            name: _sandbox_profile_template_settings(profile)
            for name, profile in cast(
                Mapping[str, Mapping[str, object]],
                sandbox_config["profiles"],
            ).items()
        },
    }
    if getattr(args, "tool_shell_sandbox_profile", None) is None:
        rendered["default_profile"] = sandbox_config["default_profile"]
    return rendered


def _shell_sandbox_template_settings(
    args: Namespace,
) -> dict[str, object] | None:
    rendered: dict[str, object] = {}
    profile = getattr(args, "tool_shell_sandbox_profile", None)
    if profile is not None:
        rendered["profile"] = profile
    if getattr(args, "tool_shell_sandbox_required", None):
        rendered["required"] = True
    if not rendered:
        return None
    assert (
        _shell_execution_mode_from_args(args) == "sandbox"
    ), "tool.shell.sandbox requires tool.shell backend sandbox"
    sandbox_config = _agent_sandbox_config_from_args(args)
    assert sandbox_config is not None, "sandbox backend is required"
    return rendered


def _is_simple_string_mapping(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    )


def _is_simple_string_sequence(value: object) -> bool:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return False
    return all(isinstance(item, str) for item in value)


def _is_default_container_template_value(
    key: str,
    value: object,
) -> bool:
    return (
        (key == "workspace_root" and value == ".")
        or (key == "pull_policy" and value == "never")
        or (key == "platform" and value == "linux/amd64")
        or (key == "network" and value == "none")
        or (key == "review_mode" and value == "deny")
    )


def _sandbox_profile_template_settings(
    profile: Mapping[str, object],
) -> dict[str, object]:
    rendered: dict[str, object] = {}
    for key, value in profile.items():
        if key == "network":
            network = _sandbox_network_template_settings(
                cast(Mapping[str, object], value)
            )
            if network:
                rendered[key] = network
        elif key == "output":
            output = _sandbox_output_template_settings(
                cast(Mapping[str, object], value)
            )
            if output:
                rendered[key] = output
        elif key == "resources":
            if value:
                rendered[key] = value
        elif not _is_default_sandbox_template_value(key, value):
            rendered[key] = value
    return rendered


def _sandbox_network_template_settings(
    network: Mapping[str, object],
) -> dict[str, object]:
    rendered: dict[str, object] = {}
    mode = network.get("mode")
    if mode is not None and mode != "none":
        rendered["mode"] = mode
    egress_allowlist = network.get("egress_allowlist")
    if egress_allowlist:
        rendered["egress_allowlist"] = egress_allowlist
    return rendered


def _sandbox_output_template_settings(
    output: Mapping[str, object],
) -> dict[str, object]:
    rendered: dict[str, object] = {}
    for key, default in (
        ("max_stdout_bytes", 65536),
        ("max_stderr_bytes", 32768),
        ("max_artifact_bytes", 0),
        ("allow_artifacts", False),
    ):
        value = output.get(key)
        if value is not None and value != default:
            rendered[key] = value
    return rendered


def _is_default_sandbox_template_value(
    key: str,
    value: object,
) -> bool:
    return (
        (key == "child_processes" and value == "deny")
        or (key == "inherited_fds" and value == "stdio")
        or (key == "cleanup" and value == "delete")
    )


def _validate_sandbox_template_config(
    args: Namespace,
    sandbox_config: Mapping[str, object],
) -> None:
    trusted_isolation_runtime_from_mapping(
        {
            "mode": IsolationMode.SANDBOX.value,
            "sandbox": sandbox_config,
        },
        source=_cli_isolation_source(),
        selection=IsolationProfileSelection(
            mode=IsolationMode.SANDBOX,
            profile=getattr(args, "tool_shell_sandbox_profile", None),
            required=(
                bool(getattr(args, "tool_shell_sandbox_required", None))
                or _shell_execution_mode_from_args(args) == "sandbox"
            ),
        ),
    )


def _has_container_policy_args(args: Namespace) -> bool:
    return getattr(
        args, "tool_container_backend", None
    ) is not None or _has_container_profile_args(args)


def _has_sandbox_policy_args(args: Namespace) -> bool:
    return getattr(
        args, "tool_sandbox_backend", None
    ) is not None or _has_sandbox_profile_args(args)


def _require_shell_backend_for_policy(
    args: Namespace,
    shell_settings: ShellToolSettings | None,
    mode: str,
    policy_name: str,
) -> None:
    backend = (
        shell_settings.backend
        if shell_settings is not None
        else _shell_execution_mode_from_args(args)
    )
    assert backend == mode, f"{policy_name} requires tool.shell backend {mode}"


def _shell_execution_mode_from_args(args: Namespace) -> str | None:
    backend = getattr(args, "tool_shell_backend", None)
    execution_mode = getattr(args, "tool_shell_execution_mode", None)
    assert (
        backend is None or execution_mode is None or backend == execution_mode
    ), "tool shell backend and execution mode must match"
    return cast(str | None, backend or execution_mode)


def _agent_enabled_tools(args: Namespace) -> list[str] | None:
    tools = (args.tool or []) + (getattr(args, "tools", None) or [])
    if tools:
        return tools
    return None


def _agent_tool_format(args: Namespace) -> ToolFormat | None:
    if getattr(args, "tool_format", None):
        return ToolFormat(args.tool_format)
    if not _agent_enabled_tools(args):
        return None
    backend = getattr(args, "backend", None)
    if backend in (
        Backend.DS4,
        Backend.DS4.value,
        Backend.MLXLM,
        Backend.MLXLM.value,
    ):
        return ToolFormat.REACT
    return None


def _agent_tool_name_policy(args: Namespace) -> ToolNamePolicySettings | None:
    mode = getattr(args, "tool_name_policy", None)
    prefix = getattr(args, "tool_name_prefix", None)
    replacement = getattr(args, "tool_name_replacement", None)
    collapse = getattr(args, "tool_name_collapse_replacement", None)
    map_entries = getattr(args, "tool_name_map", None)
    if (
        mode is None
        and prefix is None
        and replacement is None
        and collapse is None
        and not map_entries
    ):
        return None

    name_map: dict[str, str] = {}
    for entry in map_entries or []:
        assert isinstance(entry, str)
        canonical_name, separator, provider_name = entry.partition("=")
        assert separator, "tool name map entries must be CANONICAL=PROVIDER"
        assert canonical_name.strip(), "tool name map canonical name is empty"
        assert provider_name.strip(), "tool name map provider name is empty"
        name_map[canonical_name] = provider_name

    return ToolNamePolicySettings(
        mode=(
            ToolNamePolicyMode(mode)
            if mode is not None
            else ToolNamePolicyMode.ENCODED
        ),
        prefix="avl_" if prefix is None else prefix,
        replacement="_" if replacement is None else replacement,
        collapse_replacement=True if collapse is None else collapse,
        map=name_map,
    )


def _agent_tool_name_policy_kwargs(
    args: Namespace,
) -> dict[str, Any]:
    policy = _agent_tool_name_policy(args)
    return {"tool_name_policy": policy} if policy is not None else {}


def _agent_container_runtime_settings(
    args: Namespace,
    shell_settings: ShellToolSettings | None,
) -> ContainerToolRuntimeSettings | None:
    if _has_container_policy_args(args):
        _require_shell_backend_for_policy(
            args,
            shell_settings,
            "container",
            "tool.container",
        )
    container_config = _agent_container_config_from_args(args)
    assert not (
        container_config is not None
        and shell_settings is not None
        and shell_settings.backend == "sandbox"
    ), "sandbox shell execution cannot carry container policy"
    shell_selection = _agent_shell_container_selection_from_args(
        args,
        shell_settings,
    )
    if container_config is None:
        assert (
            shell_selection is None or shell_selection.profile is None
        ), "required container profile unavailable"
        return None
    runtime = trusted_container_runtime_from_mapping(
        container_config,
        source=_cli_container_source(),
        selection=shell_selection,
    )
    return ContainerToolRuntimeSettings(
        effective_settings=runtime.effective_settings,
        backend=_agent_container_backend_from_args(args),
        opt_in_backends=_agent_container_opt_in_backends(args),
        rootful_authorized=runtime.rootful_authorized,
    )


def _agent_container_backend_from_args(
    args: Namespace,
) -> ContainerAsyncBackend | None:
    backend = getattr(args, "tool_container_backend", None)
    if backend == _DOCKER_CONTAINER_BACKEND:
        backend_cls = _docker_container_backend_class()
    elif backend == _APPLE_CONTAINER_BACKEND:
        backend_cls = _apple_container_backend_class()
    else:
        return None
    if backend_cls is None:
        return None
    container_backend = backend_cls()
    assert isinstance(container_backend, ContainerAsyncBackend)
    return container_backend


def _agent_container_opt_in_backends(args: Namespace) -> tuple[str, ...]:
    if (
        getattr(args, "tool_container_backend", None)
        == _APPLE_CONTAINER_BACKEND
    ):
        return (_APPLE_CONTAINER_BACKEND,)
    return ()


def _apple_container_backend_class() -> type[ContainerAsyncBackend] | None:
    for module_name in _APPLE_CONTAINER_BACKEND_MODULES:
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            continue
        for candidate in vars(module).values():
            if _is_apple_container_backend_class(candidate):
                return cast(type[ContainerAsyncBackend], candidate)
    return None


def _docker_container_backend_class() -> type[ContainerAsyncBackend] | None:
    for module_name in _DOCKER_CONTAINER_BACKEND_MODULES:
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            continue
        for candidate in vars(module).values():
            if _is_docker_container_backend_class(candidate):
                return cast(type[ContainerAsyncBackend], candidate)
    return None


def _is_apple_container_backend_class(candidate: object) -> bool:
    return (
        isinstance(candidate, type)
        and issubclass(candidate, ContainerAsyncBackend)
        and candidate is not ContainerAsyncBackend
        and "apple" in candidate.__name__.lower()
    )


def _is_docker_container_backend_class(candidate: object) -> bool:
    return (
        isinstance(candidate, type)
        and issubclass(candidate, ContainerAsyncBackend)
        and candidate is not ContainerAsyncBackend
        and "docker" in candidate.__name__.lower()
    )


def _agent_container_config_from_args(
    args: Namespace,
) -> dict[str, object] | None:
    backend = getattr(args, "tool_container_backend", None)
    if backend is None and not _has_container_profile_args(args):
        return None
    assert backend is not None, "container backend is required"
    assert (
        backend in _SUPPORTED_CONTAINER_BACKENDS
    ), "container backend is unsupported"
    profile_name = getattr(args, "tool_container_profile", None) or (
        getattr(args, "tool_shell_container_profile", None)
        or "workspace-readonly"
    )
    image = getattr(args, "tool_container_image", None)
    assert image is not None, "container image is required"
    profile: dict[str, object] = {
        "image": image,
        "workspace_root": (
            getattr(
                args,
                "tool_container_workspace_root",
                None,
            )
            or getattr(args, "tool_shell_workspace_root", None)
            or "."
        ),
    }
    for source_key, target_key in (
        ("tool_container_pull_policy", "pull_policy"),
        ("tool_container_platform", "platform"),
        ("tool_container_network_mode", "network"),
        ("tool_container_review_mode", "review_mode"),
    ):
        value = getattr(args, source_key, None)
        if value is not None:
            profile[target_key] = value
    resources = _agent_container_resources_from_args(args)
    if resources:
        profile["resources"] = resources
    return {
        "backend": backend,
        "default_profile": profile_name,
        "allowed_profiles": (profile_name,),
        "profiles": {profile_name: profile},
        "profile_registry_id": "default",
        "policy_version": "phase11",
    }


def _agent_container_resources_from_args(
    args: Namespace,
) -> dict[str, int]:
    resources: dict[str, int] = {}
    for source_key, target_key in (
        ("tool_container_cpu_count", "cpu_count"),
        ("tool_container_memory_bytes", "memory_bytes"),
        ("tool_container_pids", "pids"),
        ("tool_container_timeout_seconds", "timeout_seconds"),
    ):
        value = getattr(args, source_key, None)
        if value is not None:
            resources[target_key] = value
    return resources


def _agent_shell_container_selection_from_args(
    args: Namespace,
    shell_settings: ShellToolSettings | None,
) -> ContainerProfileSelection | None:
    profile = getattr(args, "tool_shell_container_profile", None)
    required = getattr(args, "tool_shell_container_required", None)
    explicit_selection = profile is not None or required is not None
    shell_backend_container = (
        shell_settings is not None and shell_settings.backend == "container"
    )
    if explicit_selection:
        assert (
            shell_backend_container
        ), "tool.shell.container requires tool.shell backend container"
    if required is None and shell_settings is not None:
        required = shell_settings.backend == "container"
    if profile is None and not required:
        return None
    return container_selection_from_mapping(
        {
            "profile": profile,
            "required": bool(required),
        },
        source=_cli_container_source(),
    )


def _agent_isolation_runtime_settings(
    args: Namespace,
    shell_settings: ShellToolSettings | None,
) -> IsolationToolRuntimeSettings | None:
    if _has_sandbox_policy_args(args):
        _require_shell_backend_for_policy(
            args,
            shell_settings,
            "sandbox",
            "tool.sandbox",
        )
    sandbox_config = _agent_sandbox_config_from_args(args)
    assert not (
        sandbox_config is not None
        and shell_settings is not None
        and shell_settings.backend == "container"
    ), "container shell execution cannot carry sandbox policy"
    shell_selection = _agent_shell_sandbox_selection_from_args(
        args,
        shell_settings,
    )
    if sandbox_config is None:
        assert (
            shell_selection is None or shell_selection.profile is None
        ), "required sandbox profile unavailable"
        return None
    runtime = trusted_isolation_runtime_from_mapping(
        {
            "mode": IsolationMode.SANDBOX.value,
            "sandbox": sandbox_config,
        },
        source=_cli_isolation_source(),
        selection=(
            None
            if shell_selection is None
            else IsolationProfileSelection(
                mode=IsolationMode.SANDBOX,
                profile=shell_selection.profile,
                required=shell_selection.required,
            )
        ),
        sandbox_backend=_agent_sandbox_backend_from_args(args),
    )
    return runtime


def _agent_sandbox_backend_from_args(
    args: Namespace,
) -> SandboxAsyncBackend | None:
    backend = getattr(args, "tool_sandbox_backend", None)
    if backend == _SEATBELT_SANDBOX_BACKEND:
        return SeatbeltSandboxBackend()
    if backend == _BUBBLEWRAP_SANDBOX_BACKEND:
        return BubblewrapSandboxBackend()
    return None


def _agent_sandbox_config_from_args(
    args: Namespace,
) -> dict[str, object] | None:
    backend = getattr(args, "tool_sandbox_backend", None)
    if backend is None and not _has_sandbox_profile_args(args):
        return None
    assert backend is not None, "sandbox backend is required"
    assert (
        backend in _SUPPORTED_SANDBOX_BACKENDS
    ), "sandbox backend is unsupported"
    profile_name = getattr(args, "tool_sandbox_profile", None) or (
        getattr(args, "tool_shell_sandbox_profile", None) or "host-tools"
    )
    profile: dict[str, object] = {}
    for source_key, target_key in (
        ("tool_sandbox_trusted_executables", "trusted_executables"),
        ("tool_sandbox_executable_search_roots", "executable_search_roots"),
        ("tool_sandbox_read_roots", "read_roots"),
        ("tool_sandbox_write_roots", "write_roots"),
        ("tool_sandbox_deny_roots", "deny_roots"),
        ("tool_sandbox_scratch_roots", "scratch_roots"),
        ("tool_sandbox_output_roots", "output_roots"),
    ):
        value = getattr(args, source_key, None)
        if value:
            profile[target_key] = tuple(value)
    network = _agent_sandbox_network_from_args(args)
    if network:
        profile["network"] = network
    resources = _agent_sandbox_resources_from_args(args)
    if resources:
        profile["resources"] = resources
    output = _agent_sandbox_output_from_args(args)
    if output:
        profile["output"] = output
    for source_key, target_key in (
        ("tool_sandbox_child_processes", "child_processes"),
        ("tool_sandbox_inherited_fds", "inherited_fds"),
    ):
        value = getattr(args, source_key, None)
        if value is not None:
            profile[target_key] = value
    return {
        "backend": backend,
        "default_profile": profile_name,
        "allowed_profiles": (profile_name,),
        "profiles": {profile_name: profile},
        "profile_registry_id": "default",
        "policy_version": "phase8",
    }


def _agent_sandbox_network_from_args(args: Namespace) -> dict[str, object]:
    network: dict[str, object] = {}
    mode = getattr(args, "tool_sandbox_network_mode", None)
    if mode is not None:
        network["mode"] = mode
    egress_allowlist = getattr(args, "tool_sandbox_network_egress", None)
    if egress_allowlist:
        network["egress_allowlist"] = tuple(egress_allowlist)
    return network


def _agent_sandbox_resources_from_args(args: Namespace) -> dict[str, int]:
    resources: dict[str, int] = {}
    for source_key, target_key in (
        ("tool_sandbox_timeout_seconds", "timeout_seconds"),
        ("tool_sandbox_pids", "pids"),
    ):
        value = getattr(args, source_key, None)
        if value is not None:
            resources[target_key] = value
    return resources


def _agent_sandbox_output_from_args(args: Namespace) -> dict[str, int | bool]:
    output: dict[str, int | bool] = {}
    for source_key, target_key in (
        ("tool_sandbox_max_stdout_bytes", "max_stdout_bytes"),
        ("tool_sandbox_max_stderr_bytes", "max_stderr_bytes"),
        ("tool_sandbox_max_artifact_bytes", "max_artifact_bytes"),
    ):
        value = getattr(args, source_key, None)
        if value is not None:
            output[target_key] = value
    allow_artifacts = getattr(args, "tool_sandbox_allow_artifacts", None)
    if allow_artifacts is not None:
        output["allow_artifacts"] = allow_artifacts
    return output


def _agent_shell_sandbox_selection_from_args(
    args: Namespace,
    shell_settings: ShellToolSettings | None,
) -> SandboxProfileSelection | None:
    profile = getattr(args, "tool_shell_sandbox_profile", None)
    required = getattr(args, "tool_shell_sandbox_required", None)
    explicit_selection = profile is not None or required is not None
    shell_backend_sandbox = (
        shell_settings is not None and shell_settings.backend == "sandbox"
    )
    if explicit_selection:
        assert (
            shell_backend_sandbox
        ), "tool.shell.sandbox requires tool.shell backend sandbox"
    if required is None and shell_settings is not None:
        required = shell_settings.backend == "sandbox"
    if profile is None and not required:
        return None
    return SandboxProfileSelection(profile=profile, required=bool(required))


def _has_container_profile_args(args: Namespace) -> bool:
    return any(
        getattr(args, key, None) is not None
        for key in (
            "tool_container_profile",
            "tool_container_image",
            "tool_container_workspace_root",
            "tool_container_pull_policy",
            "tool_container_platform",
            "tool_container_cpu_count",
            "tool_container_memory_bytes",
            "tool_container_pids",
            "tool_container_timeout_seconds",
            "tool_container_network_mode",
            "tool_container_review_mode",
        )
    )


def _has_sandbox_profile_args(args: Namespace) -> bool:
    return any(
        getattr(args, key, None) is not None
        for key in (
            "tool_sandbox_profile",
            "tool_sandbox_trusted_executables",
            "tool_sandbox_executable_search_roots",
            "tool_sandbox_read_roots",
            "tool_sandbox_write_roots",
            "tool_sandbox_deny_roots",
            "tool_sandbox_scratch_roots",
            "tool_sandbox_output_roots",
            "tool_sandbox_network_mode",
            "tool_sandbox_network_egress",
            "tool_sandbox_timeout_seconds",
            "tool_sandbox_pids",
            "tool_sandbox_max_stdout_bytes",
            "tool_sandbox_max_stderr_bytes",
            "tool_sandbox_max_artifact_bytes",
            "tool_sandbox_allow_artifacts",
            "tool_sandbox_child_processes",
            "tool_sandbox_inherited_fds",
        )
    )


def _cli_container_source() -> ContainerSettingsSource:
    return ContainerSettingsSource(
        surface=ContainerSurface.CLI,
        trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
    )


def _cli_isolation_source() -> IsolationSettingsSource:
    return trusted_isolation_source(IsolationSettingsSurface.CLI)


def _agent_skills_settings(args: Namespace) -> TrustedSkillSettings | None:
    if not _has_agent_skills_args(args):
        return None

    source_roots = _skills_assignment_map(
        getattr(args, "tool_skills_source", None),
        "--tool-skills-source",
    )
    manifest_paths = _skills_assignment_map(
        getattr(args, "tool_skills_file", None),
        "--tool-skills-file",
    )
    duplicate_source_labels = set(source_roots) & set(manifest_paths)
    assert (
        not duplicate_source_labels
    ), "skills source and file labels must be unique"
    source_authorities = _skills_source_authority_map(
        getattr(args, "tool_skills_source_authority", None),
    )
    source_packages = _skills_assignment_map(
        getattr(args, "tool_skills_source_package", None),
        "--tool-skills-source-package",
    )
    allow_hidden_labels = _skills_label_tuple(
        getattr(args, "tool_skills_source_allow_hidden", None),
        "--tool-skills-source-allow-hidden",
    )
    known_source_labels = set(source_roots) | set(manifest_paths)
    unknown_labels = (
        set(source_authorities) | set(allow_hidden_labels)
    ) - known_source_labels
    assert not unknown_labels, "skills source options reference unknown labels"
    unknown_package_labels = set(source_packages) - set(source_roots)
    assert (
        not unknown_package_labels
    ), "skills source package options reference unknown labels"

    sources = tuple(
        SkillSourceConfig(
            label=label,
            authority=source_authorities.get(
                label,
                WorkspaceSkillSourceAuthority(),
            ),
            root_path=root_path,
            package_path=source_packages.get(label),
            allow_hidden_paths=label in allow_hidden_labels,
        )
        for label, root_path in source_roots.items()
    ) + tuple(
        SkillSourceConfig(
            label=label,
            authority=source_authorities.get(
                label,
                WorkspaceSkillSourceAuthority(),
            ),
            manifest_path=manifest_path,
            allow_hidden_paths=label in allow_hidden_labels,
        )
        for label, manifest_path in manifest_paths.items()
    )

    values: dict[str, object] = {
        "enabled": not bool(getattr(args, "tool_skills_disabled", False)),
        "bootstrap_enabled": (
            getattr(args, "tool_skills_bootstrap", None) != "off"
        ),
        "bootstrap_prompt": _skills_bootstrap_prompt_settings(args),
        "manifest_auto_enable": not bool(
            getattr(args, "tool_skills_file_no_auto_enable", False)
        ),
        "sources": sources,
        "read_limits": _skills_read_limits(args),
        "index_limits": _skills_index_limits(args),
        "source_limits": _skills_source_limits(args),
        "cursor_limits": _skills_cursor_limits(args),
        "privacy": _skills_privacy_settings(args),
        "observability": _skills_observability_settings(args),
    }
    authority_kinds = _skills_authority_kind_tuple(
        getattr(args, "tool_skills_authority_kind", None)
    )
    if authority_kinds:
        values["authority_kinds"] = authority_kinds
    skill_ids = tuple(getattr(args, "tool_skills_skill", None) or ())
    if skill_ids:
        values["allowed_skill_ids"] = skill_ids
        values["allowed_skill_ids_explicit"] = True
    return TrustedSkillSettings(**cast(Any, values))


def _has_agent_skills_args(args: Namespace) -> bool:
    for name in (
        "tool_skills_source",
        "tool_skills_file",
        "tool_skills_file_no_auto_enable",
        "tool_skills_source_authority",
        "tool_skills_source_package",
        "tool_skills_source_allow_hidden",
        "tool_skills_authority_kind",
        "tool_skills_skill",
        "tool_skills_disabled",
        "tool_skills_bootstrap",
        "tool_skills_bootstrap_omit",
        "tool_skills_bootstrap_instruction",
        "tool_skills_diagnostics",
        "tool_skills_observability",
        "tool_skills_max_bytes_per_read",
        "tool_skills_max_lines_per_read",
        "tool_skills_max_skills",
        "tool_skills_max_resources_per_skill",
        "tool_skills_max_indexed_bytes",
        "tool_skills_max_sources",
        "tool_skills_max_resources_per_source",
        "tool_skills_max_source_depth",
        "tool_skills_max_files_per_source",
        "tool_skills_max_directory_entries_per_source",
        "tool_skills_max_active_cursors",
        "tool_skills_max_cursor_age_seconds",
    ):
        value = getattr(args, name, None)
        if isinstance(value, bool):
            if value:
                return True
        elif value is not None:
            return True
    return False


def _skills_bootstrap_prompt_settings(
    args: Namespace,
) -> SkillBootstrapPromptSettings:
    omitted = tuple(getattr(args, "tool_skills_bootstrap_omit", None) or ())
    unknown = sorted(set(omitted) - set(_SKILL_BOOTSTRAP_PROMPT_OMIT_FIELDS))
    assert not unknown, "--tool-skills-bootstrap-omit has unknown sections"
    values: dict[str, object] = {
        field_name: section not in omitted
        for section, field_name in _SKILL_BOOTSTRAP_PROMPT_OMIT_FIELDS.items()
    }
    instructions = tuple(
        getattr(args, "tool_skills_bootstrap_instruction", None) or ()
    )
    if instructions:
        values["additional_instructions"] = instructions
    return SkillBootstrapPromptSettings(**cast(Any, values))


def _skills_assignment_map(
    items: Sequence[str] | None,
    option_name: str,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items or ():
        label, value = _skills_assignment(item, option_name)
        assert label not in values, f"{option_name} labels must be unique"
        values[label] = value
    return values


def _skills_assignment(value: str, option_name: str) -> tuple[str, str]:
    assert isinstance(value, str), f"{option_name} must be a string"
    label, separator, item = value.partition("=")
    assert separator, f"{option_name} must use LABEL=VALUE"
    label = label.strip()
    item = item.strip()
    assert label, f"{option_name} label must be non-empty"
    assert item, f"{option_name} value must be non-empty"
    return label, item


def _skills_label_tuple(
    values: Sequence[str] | None,
    option_name: str,
) -> tuple[str, ...]:
    labels: list[str] = []
    for value in values or ():
        assert isinstance(value, str), f"{option_name} must be a string"
        label = value.strip()
        assert label, f"{option_name} label must be non-empty"
        labels.append(label)
    assert len(set(labels)) == len(
        labels
    ), f"{option_name} labels must be unique"
    return tuple(labels)


def _skills_source_authority_map(
    items: Sequence[str] | None,
) -> dict[str, SkillSourceAuthority]:
    values: dict[str, SkillSourceAuthority] = {}
    for item in items or ():
        label, value = _skills_assignment(
            item,
            "--tool-skills-source-authority",
        )
        assert (
            label not in values
        ), "--tool-skills-source-authority labels must be unique"
        values[label] = _skills_source_authority(value)
    return values


def _skills_source_authority(value: str) -> SkillSourceAuthority:
    kind_value, separator, identity = value.partition(":")
    try:
        kind = SkillSourceAuthorityKind(kind_value)
    except ValueError as exc:
        raise AssertionError("unsupported skills source authority") from exc
    if kind is SkillSourceAuthorityKind.BUNDLED:
        return BundledSkillSourceAuthority(
            bundle_id=identity if separator else "avalan"
        )
    if kind is SkillSourceAuthorityKind.WORKSPACE:
        return WorkspaceSkillSourceAuthority(
            workspace_id=identity if separator else "workspace"
        )
    if kind is SkillSourceAuthorityKind.USER_LOCAL:
        return UserLocalSkillSourceAuthority(
            profile_id=identity if separator else "user-local"
        )
    if kind is SkillSourceAuthorityKind.PLUGIN_PROVIDED:
        assert identity, "plugin_provided skills authority requires plugin id"
        return PluginProvidedSkillSourceAuthority(plugin_id=identity)
    if kind is SkillSourceAuthorityKind.PREINSTALLED_REMOTE:
        assert (
            identity
        ), "preinstalled_remote skills authority requires registry id"
        return PreinstalledRemoteSkillSourceAuthority(registry_id=identity)
    raise AssertionError("unsupported skills source authority")


def _skills_authority_kind_tuple(
    values: Sequence[str] | None,
) -> tuple[SkillSourceAuthorityKind, ...]:
    if not values:
        return ()
    return tuple(SkillSourceAuthorityKind(value) for value in values)


def _skills_read_limits(args: Namespace) -> SkillReadLimits:
    return SkillReadLimits(
        **cast(
            Any,
            _skills_values(
                args,
                {
                    "max_bytes_per_read": "tool_skills_max_bytes_per_read",
                    "max_lines_per_read": "tool_skills_max_lines_per_read",
                },
            ),
        )
    )


def _skills_index_limits(args: Namespace) -> SkillIndexLimits:
    return SkillIndexLimits(
        **cast(
            Any,
            _skills_values(
                args,
                {
                    "max_skills": "tool_skills_max_skills",
                    "max_resources_per_skill": (
                        "tool_skills_max_resources_per_skill"
                    ),
                    "max_indexed_bytes": "tool_skills_max_indexed_bytes",
                },
            ),
        )
    )


def _skills_source_limits(args: Namespace) -> SkillSourceLimits:
    return SkillSourceLimits(
        **cast(
            Any,
            _skills_values(
                args,
                {
                    "max_sources": "tool_skills_max_sources",
                    "max_resources_per_source": (
                        "tool_skills_max_resources_per_source"
                    ),
                    "max_source_depth": "tool_skills_max_source_depth",
                    "max_files_per_source": "tool_skills_max_files_per_source",
                    "max_directory_entries_per_source": (
                        "tool_skills_max_directory_entries_per_source"
                    ),
                },
            ),
        )
    )


def _skills_cursor_limits(args: Namespace) -> SkillCursorLimits:
    return SkillCursorLimits(
        **cast(
            Any,
            _skills_values(
                args,
                {
                    "max_active_cursors": "tool_skills_max_active_cursors",
                    "max_cursor_age_seconds": (
                        "tool_skills_max_cursor_age_seconds"
                    ),
                },
            ),
        )
    )


def _skills_privacy_settings(args: Namespace) -> SkillPrivacySettings:
    diagnostics = getattr(args, "tool_skills_diagnostics", None)
    return SkillPrivacySettings(
        include_diagnostic_paths=diagnostics != "off",
    )


def _skills_observability_settings(
    args: Namespace,
) -> SkillObservabilitySettings:
    observability = getattr(args, "tool_skills_observability", None)
    diagnostics = getattr(args, "tool_skills_diagnostics", None)
    values: dict[str, object] = {}
    if observability == "off":
        values.update(
            {
                "enabled": False,
                "emit_events": False,
                "include_diagnostics": False,
                "include_byte_counts": False,
            }
        )
    elif observability == "verbose":
        values["include_byte_counts"] = True
    if diagnostics == "off":
        values["include_diagnostics"] = False
    elif diagnostics == "verbose":
        values["include_diagnostics"] = True
        values["include_byte_counts"] = True
    return SkillObservabilitySettings(**cast(Any, values))


def _skills_values(
    args: Namespace,
    mapping: Mapping[str, str],
) -> dict[str, object]:
    return {
        target: value
        for target, source in mapping.items()
        if (value := getattr(args, source, None)) is not None
    }


def _agent_tool_settings(args: Namespace) -> ToolSettingsContext:
    browser_settings = get_tool_settings(
        args, prefix="browser", settings_cls=BrowserToolSettings
    )
    database_settings = get_tool_settings(
        args, prefix="database", settings_cls=DatabaseToolSettings
    )
    graph_settings = get_tool_settings(
        args, prefix="graph", settings_cls=GraphToolSettings
    )
    shell_settings = get_tool_settings(
        args, prefix="shell", settings_cls=ShellToolSettings
    )
    shell_explicit_fields = _tool_settings_explicit_fields_from_mapping(
        args,
        prefix="shell",
        settings_cls=ShellToolSettings,
    )
    container_runtime = _agent_container_runtime_settings(
        args,
        shell_settings,
    )
    return ToolSettingsContext(
        browser=browser_settings,
        database=database_settings,
        graph=graph_settings,
        skills=_agent_skills_settings(args),
        shell=shell_settings,
        shell_explicit_fields=(
            shell_explicit_fields if shell_settings is not None else None
        ),
        container=container_runtime,
        isolation=_agent_isolation_runtime_settings(args, shell_settings),
    )


def _agent_server_output_redaction_settings(
    args: Namespace,
) -> ServerOutputRedactionSettings | None:
    enabled = getattr(args, "server_output_redaction_enabled", None)
    rules = getattr(args, "server_output_redaction_rules", None)
    protocols = getattr(args, "server_output_redaction_protocols", None)
    channels = getattr(args, "server_output_redaction_channels", None)
    if enabled is None and not rules and not protocols and not channels:
        return None

    values: dict[str, object] = {
        "enabled": bool(enabled or rules or protocols or channels)
    }
    if rules:
        values["rules"] = frozenset(
            cast(Iterable[ServerOutputRedactionRule], rules)
        )
    if protocols:
        values["protocols"] = frozenset(
            cast(Iterable[ServerOutputRedactionProtocol], protocols)
        )
    if channels:
        values["channels"] = frozenset(
            cast(Iterable[ServerOutputRedactionChannel], channels)
        )
    return ServerOutputRedactionSettings(**cast(Any, values))


async def _agent_run_input(
    input_string: str | None, file_paths: list[str] | None
) -> Message | str | None:
    """Build agent run input from text and local file paths."""
    return await input_files(input_string, file_paths)


def _uses_ds4_backend(orchestrator: Orchestrator) -> bool:
    """Return whether the active orchestrator engine uses DS4."""
    engine_agent = orchestrator.engine_agent
    if not engine_agent:
        return False

    engine_uri = getattr(engine_agent, "engine_uri", None)
    params = getattr(engine_uri, "params", None)
    if isinstance(params, dict):
        backend = params.get("backend")
        if backend == Backend.DS4 or backend == Backend.DS4.value:
            return True

    engine = orchestrator.engine
    model_type = getattr(engine, "model_type", None)
    return isinstance(model_type, str) and model_type.lower().startswith("ds4")


def _agent_display_models(
    orchestrator: Orchestrator,
    hub: HuggingfaceHub,
    *,
    is_local: bool,
) -> list[Model | str]:
    """Return model display payloads without querying DS4 local paths."""
    if not is_local or _uses_ds4_backend(orchestrator):
        return list(orchestrator.model_ids)

    return [hub.model(model_id) for model_id in orchestrator.model_ids]


async def agent_message_search(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: HuggingfaceHub,
    logger: Logger,
    refresh_per_second: int,
) -> None:
    _, _i = theme._, theme.icons

    specs_path = args.specifications_file
    engine_uri = getattr(args, "engine_uri", None)
    assert not (
        specs_path and engine_uri
    ), "specifications file and --engine-uri are mutually exclusive"
    assert (
        specs_path or engine_uri
    ), "specifications file or --engine-uri must be specified"
    agent_id = args.id
    participant_id = args.participant
    session_id = args.session

    assert agent_id and participant_id and session_id

    tty_path = getattr(args, "tty", "/dev/tty") or "/dev/tty"

    input_string = get_input(
        console,
        _i["user_input"] + " ",
        echo_stdin=not args.no_repl,
        is_quiet=args.quiet,
        tty_path=tty_path,
    )
    if not input_string:
        return

    limit = args.limit

    async with AsyncExitStack() as stack:
        loader = OrchestratorLoader(
            hub=hub,
            logger=logger,
            participant_id=participant_id,
            stack=stack,
        )
        with console.status(
            _("Loading agent..."),
            spinner=theme.get_spinner("agent_loading") or "dots",
            refresh_per_second=refresh_per_second,
        ):
            if specs_path:
                logger.debug(
                    "Loading agent from %s for participant %s",
                    specs_path,
                    participant_id,
                )

                orchestrator = await loader.from_file(
                    specs_path,
                    agent_id=agent_id,
                    tool_settings=_agent_tool_settings(args),
                    **_agent_tool_name_policy_kwargs(args),
                    event_manager_mode=EventManagerMode.CLI,
                )
            else:
                assert (
                    args.engine_uri
                ), "--engine-uri required when no specifications file"
                logger.debug("Loading agent from inline settings")
                tool_settings = _agent_tool_settings(args)
                memory_recent = (
                    args.memory_recent
                    if args.memory_recent is not None
                    else True
                )
                settings = get_orchestrator_settings(
                    args,
                    agent_id=agent_id,
                    memory_recent=memory_recent,
                    tools=_agent_enabled_tools(args),
                )
                orchestrator = await loader.from_settings(
                    settings,
                    tool_settings=tool_settings,
                    **_agent_tool_name_policy_kwargs(args),
                    event_manager_mode=EventManagerMode.CLI,
                )
            orchestrator = await stack.enter_async_context(orchestrator)

            assert orchestrator.engine_agent
            assert orchestrator.engine and orchestrator.engine.model_id

            is_local = not isinstance(
                orchestrator.engine, TextGenerationVendorModel
            )
            can_access = (
                True
                if _uses_ds4_backend(orchestrator)
                else args.skip_hub_access_check
                or hub.can_access(orchestrator.engine.model_id)
            )
            models = _agent_display_models(
                orchestrator, hub, is_local=is_local
            )

            console.print(
                theme.agent(orchestrator, models=models, can_access=can_access)
            )

            logger.debug(
                'Searching for "%s" across messages on session %s between '
                "agent %s and participant %s",
                input_string,
                session_id,
                agent_id,
                participant_id,
            )
            messages = await orchestrator.memory.search_messages(
                search=input_string,
                agent_id=agent_id,
                search_user_messages=False,
                session_id=session_id,
                participant_id=participant_id,
                function=args.function,
                limit=limit,
            )
            console.print(
                theme.search_message_matches(
                    participant_id,
                    orchestrator,
                    cast(list[EngineMessageScored], messages),
                )
            )


async def agent_run(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: HuggingfaceHub,
    logger: Logger,
    refresh_per_second: int,
) -> None:
    _, _i = theme._, theme.icons
    display_config = cli_stream_display_config(
        args,
        refresh_per_second=refresh_per_second,
        interactive=bool(getattr(console, "is_terminal", True)),
    )

    specs_path = args.specifications_file
    engine_uri = getattr(args, "engine_uri", None)
    assert not (
        specs_path and engine_uri
    ), "specifications file and --engine-uri are mutually exclusive"
    assert (
        specs_path or engine_uri
    ), "specifications file or --engine-uri must be specified"
    use_async_generator = not args.use_sync_generator
    display_tokens = display_config.display_tokens
    dtokens_pick = 10 if display_tokens > 0 else 0
    with_stats = display_config.show_stats
    agent_id = args.id
    participant_id = args.participant
    session_id = args.session if not args.no_session else None
    load_recent_messages = (
        not args.skip_load_recent_messages and not args.no_session
    )
    load_recent_messages_limit = args.load_recent_messages_limit

    event_stats = EventStats()
    tty_path = getattr(args, "tty", "/dev/tty") or "/dev/tty"
    coordinator_container: dict[str, CliStreamCoordinator | None] = {
        "coordinator": None
    }

    async def _confirm_call(call: ToolCall) -> str:
        coordinator = coordinator_container["coordinator"]
        if coordinator is None:
            async with CliStreamCoordinator(
                console,
                display_config,
            ) as fallback_coordinator:
                return await fallback_coordinator.confirm_tool_call(
                    call,
                    tty_path=tty_path,
                )
        return await coordinator.confirm_tool_call(call, tty_path=tty_path)

    async def _event_listener(event: object) -> None:
        nonlocal event_stats
        event_type = getattr(event, "type", None)
        assert isinstance(event_type, EventType)
        event_stats.record_trigger(event_type)

    async def _init_orchestrator() -> Orchestrator:
        loader = OrchestratorLoader(
            hub=hub,
            logger=logger,
            participant_id=participant_id,
            stack=stack,
        )
        if specs_path:
            logger.debug(
                "Loading agent from %s for participant %s",
                specs_path,
                participant_id,
            )

            orchestrator = await loader.from_file(
                specs_path,
                agent_id=agent_id,
                disable_memory=args.no_session,
                tool_settings=_agent_tool_settings(args),
                **_agent_tool_name_policy_kwargs(args),
                event_manager_mode=EventManagerMode.CLI,
            )
        else:
            assert (
                args.engine_uri
            ), "--engine-uri required when no specifications file"
            assert not args.specifications_file or not args.engine_uri
            tool_settings = _agent_tool_settings(args)
            memory_recent = (
                args.memory_recent
                if args.memory_recent is not None
                else not args.no_session
            )
            settings = get_orchestrator_settings(
                args,
                agent_id=agent_id or uuid4(),
                memory_recent=memory_recent,
                tools=_agent_enabled_tools(args),
                max_new_tokens=getattr(args, "run_max_new_tokens", None),
                temperature=getattr(args, "run_temperature", None),
                top_k=getattr(args, "run_top_k", None),
                top_p=getattr(args, "run_top_p", None),
                use_cache=getattr(args, "run_use_cache", None),
                cache_strategy=getattr(args, "run_cache_strategy", None),
            )
            logger.debug("Loading agent from inline settings")
            tool_format = _agent_tool_format(args)
            tool_name_policy_kwargs = _agent_tool_name_policy_kwargs(args)
            tool_recovery_formats = [
                ToolCallRecoveryFormat(value)
                for value in getattr(args, "tool_recovery_format", None) or []
            ]
            if tool_recovery_formats:
                orchestrator = await loader.from_settings(
                    settings,
                    tool_settings=tool_settings,
                    tool_format=tool_format,
                    **tool_name_policy_kwargs,
                    tool_recovery_formats=tool_recovery_formats,
                    event_manager_mode=EventManagerMode.CLI,
                )
            else:
                orchestrator = await loader.from_settings(
                    settings,
                    tool_settings=tool_settings,
                    tool_format=tool_format,
                    **tool_name_policy_kwargs,
                    event_manager_mode=EventManagerMode.CLI,
                )
        event_manager = orchestrator.event_manager

        def event_manager_method(name: str) -> Any | None:
            method = getattr(event_manager, name, None)
            has_method = callable(
                getattr(type(event_manager), name, None)
            ) or name in getattr(event_manager, "__dict__", {})
            return method if has_method and callable(method) else None

        add_ui_listener = event_manager_method("add_ui_listener")
        if add_ui_listener is not None:
            add_ui_listener(_event_listener)
        else:
            add_listener = event_manager_method("add_listener")
            assert callable(add_listener)
            add_listener(_event_listener)
        remove_listener = event_manager_method("remove_listener")
        register_cleanup = getattr(stack, "callback", None)
        has_cleanup = callable(
            getattr(type(stack), "callback", None)
        ) or "callback" in getattr(stack, "__dict__", {})
        if remove_listener is not None and has_cleanup:
            assert callable(register_cleanup)
            register_cleanup(remove_listener, _event_listener)

        orchestrator = await stack.enter_async_context(orchestrator)

        if args.tools_confirm:
            assert (
                not orchestrator.tool.is_empty
            ), "--tools-confirm requires tools"

        logger.debug(
            "Agent loaded from %s, models used: %s, with recent message "
            "memory: %s, with permanent message memory: %s",
            specs_path,
            orchestrator.model_ids,
            "yes" if orchestrator.memory.has_recent_message else "no",
            (
                "yes, with session #"
                + str(orchestrator.memory.permanent_message.session_id)
                if (
                    orchestrator.memory.has_permanent_message
                    and orchestrator.memory.permanent_message is not None
                )
                else "no"
            ),
        )

        if not display_config.answer_stdout_only:
            assert orchestrator.engine_agent
            assert orchestrator.engine and orchestrator.engine.model_id

            is_local = not isinstance(
                orchestrator.engine, TextGenerationVendorModel
            )

            can_access = (
                True
                if _uses_ds4_backend(orchestrator)
                else args.skip_hub_access_check
                or not is_local
                or hub.can_access(orchestrator.engine.model_id)
            )
            models = _agent_display_models(
                orchestrator, hub, is_local=is_local
            )

            console.print(
                theme.agent(orchestrator, models=models, can_access=can_access)
            )

        if not args.no_session:
            if session_id:
                await orchestrator.memory.continue_session(
                    session_id=session_id,
                    load_recent_messages=load_recent_messages,
                    load_recent_messages_limit=load_recent_messages_limit,
                )
            else:
                await orchestrator.memory.start_session()

        if (
            load_recent_messages
            and orchestrator.memory.has_recent_message
            and not display_config.answer_stdout_only
        ):
            recent_message = orchestrator.memory.recent_message
            assert recent_message is not None
            if recent_message.is_empty:
                return orchestrator
            console.print(
                theme.recent_messages(
                    participant_id,
                    orchestrator,
                    recent_message.data,
                )
            )

        return orchestrator

    async with AsyncExitStack() as stack:
        if display_config.answer_stdout_only:
            orchestrator = await _init_orchestrator()
        else:
            with console.status(
                _("Loading agent..."),
                spinner=theme.get_spinner("agent_loading") or "dots",
                refresh_per_second=display_config.refresh_per_second,
            ):
                orchestrator = await _init_orchestrator()

        watch_spec = bool(specs_path and args.conversation and args.watch)
        if watch_spec:
            specs_mtime = getmtime(specs_path)

        input_string: str | None = None
        input_file_paths = cast(
            list[str] | None, getattr(args, "input_file", None)
        )
        in_conversation = False
        while not input_string or in_conversation:
            current_input_file_paths = (
                None if in_conversation else input_file_paths
            )
            if watch_spec and not has_input(console):
                new_mtime = getmtime(specs_path)
                if new_mtime != specs_mtime:
                    logger.debug("Reloading agent from %s", specs_path)
                    orchestrator = await _init_orchestrator()
                    specs_mtime = new_mtime
                    in_conversation = False
                    continue
            logger.debug(
                "Waiting for new message to add to orchestrator's existing "
                + str(orchestrator.memory.recent_message.size)
                if orchestrator.memory
                and orchestrator.memory.has_recent_message
                and orchestrator.memory.recent_message is not None
                else "0" + " messages"
            )
            input_string = get_input(
                console,
                _i["user_input"] + " ",
                echo_stdin=not args.no_repl,
                force_prompt=in_conversation,
                is_quiet=display_config.answer_stdout_only,
                tty_path=tty_path,
            )
            if not input_string and not current_input_file_paths:
                logger.debug("Finishing session with orchestrator")
                return

            logger.debug('Agent about to process input "%s"', input_string)
            agent_input = await _agent_run_input(
                input_string, current_input_file_paths
            )
            assert agent_input is not None
            output = await orchestrator(
                agent_input,
                use_async_generator=use_async_generator,
                tool_confirm=_confirm_call if args.tools_confirm else None,
            )

            if (
                not display_config.answer_stdout_only
                and not display_config.show_stats
                and not (
                    isinstance(theme, Theme) and theme.prefix_stream_answers
                )
            ):
                console.print(_i["agent_output"] + " ", end="")

            assert isinstance(output, OrchestratorResponse)
            assert orchestrator.engine is not None
            text_output = cast(TextGenerationResponse, output)
            text_engine = cast(TextGenerationModel, orchestrator.engine)

            await token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=orchestrator,
                event_stats=event_stats,
                lm=text_engine,
                input_string=input_string or "",
                refresh_per_second=display_config.refresh_per_second,
                response=text_output,
                dtokens_pick=dtokens_pick,
                display_tokens=display_tokens,
                tool_events_limit=display_config.display_tools_events,
                with_stats=with_stats,
                coordinator_container=coordinator_container,
                display_config=display_config,
                answer_prefix=(
                    _i["agent_output"] + " "
                    if isinstance(theme, Theme)
                    and theme.prefix_stream_answers
                    and not display_config.answer_stdout_only
                    else None
                ),
            )

            if args.conversation:
                console.print("")
                if not in_conversation:
                    in_conversation = True
            else:
                break


async def agent_serve(
    args: Namespace,
    hub: HuggingfaceHub,
    logger: Logger,
    name: str,
    version: str,
) -> None:
    assert args.host and args.port
    specs_path = args.specifications_file
    agent_id = getattr(args, "id", None)
    participant_id = args.participant
    engine_uri = getattr(args, "engine_uri", None)
    assert not (
        specs_path and engine_uri
    ), "specifications file and --engine-uri are mutually exclusive"
    assert (
        specs_path or engine_uri
    ), "specifications file or --engine-uri must be specified"

    settings: OrchestratorSettings | None = None
    tool_settings = _agent_tool_settings(args)

    protocols = await OrchestratorLoader.resolve_serve_protocols(
        specs_path=specs_path,
        cli_protocols=getattr(args, "protocol", None),
    )
    output_redaction_settings = _agent_server_output_redaction_settings(args)

    if not specs_path:
        memory_recent = (
            args.memory_recent if args.memory_recent is not None else True
        )
        settings = get_orchestrator_settings(
            args,
            agent_id=agent_id or uuid4(),
            memory_recent=memory_recent,
            tools=_agent_enabled_tools(args),
        )

    server_kwargs: dict[str, Any] = {
        "hub": hub,
        "name": name,
        "version": version,
        "mcp_prefix": getattr(args, "mcp_prefix", "/mcp") or "/mcp",
        "openai_prefix": args.openai_prefix,
        "a2a_prefix": getattr(args, "a2a_prefix", "/a2a") or "/a2a",
        "mcp_name": getattr(args, "mcp_name", "run") or "run",
        "mcp_description": getattr(args, "mcp_description", None),
        "a2a_tool_name": getattr(args, "a2a_name", "run") or "run",
        "a2a_tool_description": getattr(args, "a2a_description", None),
        "specs_path": specs_path,
        "settings": settings,
        "tool_settings": tool_settings,
        "tool_name_policy": _agent_tool_name_policy(args),
        "host": args.host,
        "port": args.port,
        "reload": args.reload,
        "logger": logger,
        "agent_id": agent_id,
        "participant_id": participant_id,
        "allow_origins": args.cors_origin,
        "allow_origin_regex": args.cors_origin_regex,
        "allow_methods": args.cors_method,
        "allow_headers": args.cors_header,
        "allow_credentials": args.cors_credentials,
        "protocols": protocols,
    }
    if output_redaction_settings is not None:
        server_kwargs["output_redaction_settings"] = output_redaction_settings
    server = agents_server(**server_kwargs)
    await server.serve()


async def agent_proxy(
    args: Namespace,
    hub: HuggingfaceHub,
    logger: Logger,
    name: str,
    version: str,
) -> None:
    args.name = getattr(args, "name", "Proxy") or "Proxy"
    args.memory_recent = getattr(args, "memory_recent", True) or True
    args.memory_permanent_message = (
        getattr(args, "memory_permanent_message", None)
        or "postgresql://avalan:password@localhost:5432/avalan"
    )
    args.specifications_file = None

    assert getattr(args, "engine_uri", None), "--engine-uri is required"

    await agent_serve(args, hub, logger, name, version)


async def agent_init(args: Namespace, console: Console, theme: Theme) -> None:
    _ = theme._
    tty_path = getattr(args, "tty", "/dev/tty") or "/dev/tty"

    name = args.name or Prompt.ask(_("Agent name"))
    role = args.role or get_input(
        console,
        _("Agent role") + " ",
        echo_stdin=not args.no_repl,
        is_quiet=args.quiet,
        tty_path=tty_path,
    )

    task = args.task or get_input(
        console,
        _("Agent task") + " ",
        echo_stdin=not args.no_repl,
        is_quiet=args.quiet,
        tty_path=tty_path,
    )
    goal_instructions = getattr(args, "goal_instructions", None) or get_input(
        console,
        _("Agent goal instructions") + " ",
        echo_stdin=not args.no_repl,
        is_quiet=args.quiet,
        tty_path=tty_path,
    )

    memory_recent = (
        args.memory_recent
        if args.memory_recent is not None
        else Confirm.ask(_("Use recent message memory?"))
    )
    memory_permanent_message = (
        args.memory_permanent_message
        if args.memory_permanent_message is not None
        else Prompt.ask(_("Permanent memory DSN"), default="")
    )
    engine_uri = args.engine_uri or Prompt.ask(
        _("Engine URI"),
        default="microsoft/Phi-4-mini-instruct",
    )

    settings = get_orchestrator_settings(
        args,
        agent_id=uuid4(),
        name=name,
        role=role,
        task=task,
        instructions=getattr(args, "instructions", None),
        goal_instructions=goal_instructions,
        engine_uri=engine_uri,
        memory_recent=memory_recent,
        memory_permanent_message=memory_permanent_message,
        max_new_tokens=(
            args.run_max_new_tokens
            if args.run_max_new_tokens is not None
            else 1024
        ),
        tools=(args.tool or []) + (getattr(args, "tools", None) or []),
        temperature=getattr(args, "run_temperature", None),
        top_k=getattr(args, "run_top_k", None),
        top_p=getattr(args, "run_top_p", None),
        use_cache=getattr(args, "run_use_cache", None),
        cache_strategy=getattr(args, "run_cache_strategy", None),
    )

    browser_tool = get_tool_settings(
        args,
        prefix="browser",
        settings_cls=BrowserToolSettings,
        open_files=False,
    )
    database_tool = get_tool_settings(
        args,
        prefix="database",
        settings_cls=DatabaseToolSettings,
        open_files=False,
    )
    graph_tool = get_tool_settings(
        args,
        prefix="graph",
        settings_cls=GraphToolSettings,
        open_files=False,
    )
    shell_tool = get_tool_settings(
        args,
        prefix="shell",
        settings_cls=ShellToolSettings,
        open_files=False,
    )
    skills_tool = _skills_tool_template_settings(_agent_skills_settings(args))

    env = Environment(
        loader=FileSystemLoader(
            join(dirname(__file__), "..", "..", "agent", "templates")
        ),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["toml_value"] = _toml_template_value
    template = env.get_template("blueprint.toml")
    tool_format = getattr(args, "tool_format", None)
    tool_recovery_formats = getattr(args, "tool_recovery_format", None) or []
    tool_name_policy = _agent_tool_name_policy(args)
    rendered = template.render(
        orchestrator=settings,
        browser_tool=browser_tool,
        database_tool=database_tool,
        graph_tool=graph_tool,
        skills_tool=skills_tool,
        shell_tool=_shell_tool_template_settings(shell_tool),
        container_tool=_container_tool_template_settings(args),
        sandbox_tool=_sandbox_tool_template_settings(args),
        shell_container=_shell_container_template_settings(args),
        shell_sandbox=_shell_sandbox_template_settings(args),
        tool_format=tool_format,
        tool_name_policy=tool_name_policy,
        tool_recovery_formats=tool_recovery_formats,
    )
    console.print(Syntax(rendered, "toml"))
