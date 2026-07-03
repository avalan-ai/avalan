from ..agent.orchestrator import Orchestrator
from ..agent.orchestrator.orchestrators.default import DefaultOrchestrator
from ..agent.orchestrator.orchestrators.json import JsonOrchestrator, Property
from ..container import (
    ContainerBackend,
    ContainerExecutionScope,
    ContainerNormalizedRuntimeEnvelopePlan,
    ContainerPlanRequest,
    ContainerPlanRequestKind,
    ContainerProfileSelection,
    ContainerRuntimeEnvelopeKind,
    ContainerSettings,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerToolRuntimeSettings,
    container_selection_from_mapping,
    normalize_runtime_envelope_plan,
    trusted_container_settings_from_mapping,
    trusted_container_source,
)
from ..entities import (
    EngineUri,
    OrchestratorSettings,
    PermanentMemoryStoreSettings,
    ToolCallRecoveryFormat,
    ToolFormat,
    ToolManagerSettings,
    ToolNamePolicyMode,
    ToolNamePolicySettings,
    TransformerEngineSettings,
)
from ..event import Event, EventType
from ..event.manager import EventManager, EventManagerMode
from ..filesystem import read_text
from ..isolation import (
    IsolationMode,
    IsolationProfileSelection,
    IsolationSettings,
    IsolationSettingsSource,
    IsolationSettingsSurface,
    IsolationToolRuntimeSettings,
    SandboxProfileSelection,
    SandboxSettings,
    trusted_isolation_source,
)
from ..memory.manager import MemoryManager
from ..model.file_delivery import LocalFileDeliveryProfile
from ..model.hubs.huggingface import HuggingfaceHub
from ..model.manager import ModelManager
from ..skill import (
    CANONICAL_SKILLS_TOOL_NAMES,
    BundledSkillSourceAuthority,
    PluginProvidedSkillSourceAuthority,
    PreinstalledRemoteSkillSourceAuthority,
    SkillConfiguredSource,
    SkillSettingsSurface,
    SkillSourceAuthority,
    SkillSourceAuthorityKind,
    SkillSourceConfig,
    TrustedSkillSettings,
    UntrustedSkillSettings,
    UserLocalSkillSourceAuthority,
    WorkspaceSkillSourceAuthority,
    build_skill_registry,
    merge_skill_settings,
    parse_untrusted_skill_settings_config,
    replace_trusted_skill_settings,
    resolve_skill_sources,
)
from ..skill.observability import skill_audit_correlation_id
from ..task.schema import TaskSchemaResolutionError, resolve_schema_ref
from ..tool import ToolSet
from ..tool.a2a import A2AToolSet
from ..tool.browser import (
    HAS_BROWSER_DEPENDENCIES,
    BrowserToolSet,
    BrowserToolSettings,
)
from ..tool.code import HAS_CODE_DEPENDENCIES, CodeToolSet
from ..tool.context import ToolSettingsContext
from ..tool.database.settings import DatabaseToolSettings
from ..tool.database.toolset import DatabaseToolSet
from ..tool.graph import (
    HAS_GRAPH_DEPENDENCIES,
    GraphToolSet,
)
from ..tool.graph_settings import GraphToolSettings
from ..tool.manager import ToolManager
from ..tool.math import MathToolSet
from ..tool.mcp import McpToolSet
from ..tool.memory import MemoryToolSet
from ..tool.names import matches_tool_namespace
from ..tool.shell import (
    ShellToolSet,
    ShellToolSettings,
    normalize_shell_enabled_tools,
    should_append_shell_toolset,
)
from ..tool.shell.input_files import shell_input_file_filter
from ..tool.shell.opt_in import enables_shell_pipeline
from ..tool.skills import SkillsToolSet

from contextlib import AsyncExitStack
from dataclasses import dataclass, fields, replace
from importlib import import_module
from logging import DEBUG, INFO, Logger
from os import R_OK, access
from os.path import exists
from tomllib import loads as toml_loads
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, cast
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from ..filters import Partitioner
else:
    Partitioner = Any

_SHELL_EXECUTION_MODE_FIELDS = frozenset({"backend", "execution_mode"})

SentenceTransformerModel: type[Any] | None = None
TextPartitioner: type[Any] | None = None
PgsqlRawMemory: type[Any] | None = None


def should_append_mcp_toolset(enabled_tools: list[str] | None) -> bool:
    """Return whether MCP tools were explicitly enabled."""
    if not enabled_tools:
        return False
    return any(
        matches_tool_namespace("mcp.call", enabled)
        for enabled in enabled_tools
    )


def should_append_a2a_toolset(enabled_tools: list[str] | None) -> bool:
    """Return whether A2A tools were explicitly enabled."""
    if not enabled_tools:
        return False
    return any(
        matches_tool_namespace("a2a.call", enabled)
        for enabled in enabled_tools
    )


def should_append_skills_toolset(
    skills_settings: TrustedSkillSettings | None,
    enabled_tools: list[str] | None,
) -> bool:
    """Return whether configured skills should be exposed as tools."""
    return (
        skills_settings is not None
        and skills_settings.enabled
        and (
            _skills_tools_requested(enabled_tools)
            or _skills_manifest_tools_auto_enabled(skills_settings)
        )
    )


def effective_skills_enabled_tools(
    skills_settings: TrustedSkillSettings | None,
    enabled_tools: list[str] | None,
) -> list[str] | None:
    """Return tool filters with auto-enabled skills tools included."""
    if not _skills_manifest_tools_auto_enabled(skills_settings):
        return enabled_tools
    if enabled_tools is None:
        return None
    if _skills_tools_requested(enabled_tools):
        return enabled_tools
    return [*enabled_tools, "skills"]


def _skills_tools_requested(enabled_tools: list[str] | None) -> bool:
    """Return whether the tool list explicitly selects skills tools."""
    if not enabled_tools:
        return False
    return any(
        matches_tool_namespace(tool_name, enabled)
        for enabled in enabled_tools
        for tool_name in CANONICAL_SKILLS_TOOL_NAMES
    )


def _skills_manifest_tools_auto_enabled(
    skills_settings: TrustedSkillSettings | None,
) -> bool:
    """Return whether manifest file sources should expose skills tools."""
    return (
        skills_settings is not None
        and skills_settings.enabled
        and skills_settings.manifest_auto_enable
        and any(
            source.enabled and source.manifest_path is not None
            for source in skills_settings.sources
        )
    )


def _skill_configured_sources(
    skills_settings: TrustedSkillSettings,
) -> tuple[SkillConfiguredSource, ...]:
    """Return resolver source configs from trusted skill settings."""
    assert isinstance(skills_settings, TrustedSkillSettings)
    sources: list[SkillConfiguredSource] = []
    for source in skills_settings.sources:
        assert isinstance(source, SkillSourceConfig)
        if source.root_path is None and source.manifest_path is None:
            continue
        sources.append(
            SkillConfiguredSource(
                label=source.label,
                authority=source.authority,
                root_path=source.root_path,
                manifest_path=source.manifest_path,
                package_path=source.package_path,
                enabled=source.enabled,
                allow_hidden_paths=source.allow_hidden_paths,
            )
        )
    return tuple(sources)


async def _build_skills_toolset(
    skills_settings: TrustedSkillSettings,
    *,
    event_manager: EventManager | None = None,
) -> SkillsToolSet:
    """Build a skills toolset from trusted settings."""
    sources = _skill_configured_sources(skills_settings)
    assert sources, "skills require at least one trusted source"
    audit_operation_id = skill_audit_correlation_id("skill-registry-build")
    source_result = await resolve_skill_sources(
        sources,
        settings=skills_settings,
        event_manager=event_manager,
        audit_operation_id=audit_operation_id,
    )
    registry = await build_skill_registry(
        source_result,
        settings=skills_settings,
        event_manager=event_manager,
        audit_operation_id=audit_operation_id,
    )
    return SkillsToolSet(
        registry,
        bootstrap_enabled=skills_settings.bootstrap_enabled,
        bootstrap_prompt_settings=skills_settings.bootstrap_prompt,
        event_manager=event_manager,
        namespace="skills",
    )


def _toolset_exposes_enabled_tool(
    toolset: ToolSet,
    enabled_tools: list[str] | None,
) -> bool:
    """Return whether a toolset retains at least one enabled tool."""
    if enabled_tools is None:
        return bool(toolset.tools)

    tool_prefix = f"{toolset.namespace}." if toolset.namespace else ""
    for tool in toolset.tools:
        name = getattr(tool, "__name__", tool.__class__.__name__)
        canonical_name = f"{tool_prefix}{name}"
        if any(
            matches_tool_namespace(canonical_name, enabled)
            for enabled in enabled_tools
        ):
            return True

    return False


def _merge_shell_tool_settings(
    base: ShellToolSettings | None,
    override: ShellToolSettings | None,
    *,
    explicit_fields: frozenset[str] | None = None,
) -> ShellToolSettings | None:
    """Return shell settings with explicit overrides applied."""
    if override is None:
        return base
    if base is None or explicit_fields is None:
        return override

    if not explicit_fields:
        return base

    valid_fields = {field.name for field in fields(ShellToolSettings)}
    assert explicit_fields <= valid_fields
    values: dict[str, object] = {}
    execution_mode = _explicit_shell_execution_mode(
        override,
        explicit_fields,
    )
    if execution_mode is not None:
        values["backend"] = execution_mode
        values["execution_mode"] = execution_mode
        if (
            execution_mode != "container"
            and "container" not in explicit_fields
        ):
            values["container"] = None
        if execution_mode != "sandbox" and "sandbox" not in explicit_fields:
            values["sandbox"] = None

    for name in explicit_fields:
        if name in _SHELL_EXECUTION_MODE_FIELDS:
            continue
        values[name] = getattr(override, name)

    return replace(base, **cast(Any, values))


def _explicit_shell_execution_mode(
    override: ShellToolSettings | None,
    explicit_fields: frozenset[str] | None,
) -> str | None:
    """Return explicitly requested shell execution mode, if any."""
    if (
        override is None
        or explicit_fields is None
        or not explicit_fields.intersection(_SHELL_EXECUTION_MODE_FIELDS)
    ):
        return None

    execution_mode = override.execution_mode
    assert isinstance(execution_mode, str)
    return execution_mode


def _shell_tool_runtime_settings(
    shell_settings: ShellToolSettings,
    container_runtime: ContainerToolRuntimeSettings | None,
    isolation_runtime: IsolationToolRuntimeSettings | None,
) -> tuple[
    ContainerToolRuntimeSettings | None,
    IsolationToolRuntimeSettings | None,
]:
    """Return runtimes compatible with the active shell backend."""
    assert isinstance(shell_settings, ShellToolSettings)
    execution_mode = shell_settings.execution_mode
    if execution_mode == "container":
        if (
            isolation_runtime is not None
            and isolation_runtime.mode is IsolationMode.CONTAINER
        ):
            return None, isolation_runtime
        return container_runtime, None
    if (
        execution_mode == "sandbox"
        and isolation_runtime is not None
        and isolation_runtime.mode is IsolationMode.SANDBOX
    ):
        return None, isolation_runtime
    return None, None


class OrchestratorLoader:
    DEFAULT_SENTENCE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_SENTENCE_MODEL_MAX_TOKENS = 500
    DEFAULT_SENTENCE_MODEL_OVERLAP_SIZE = 125
    DEFAULT_SENTENCE_MODEL_WINDOW_SIZE = 250

    _ALLOWED_PROTOCOLS = frozenset({"a2a", "flow", "mcp", "openai"})
    _OPENAI_COMPLETION_ALIASES = frozenset(
        {
            "chat",
            "completion",
            "completions",
        }
    )
    _OPENAI_ENDPOINT_COMPLETIONS = "completions"
    _OPENAI_ENDPOINT_RESPONSES = "responses"
    _OPENAI_ENDPOINTS = frozenset(
        {
            _OPENAI_ENDPOINT_COMPLETIONS,
            _OPENAI_ENDPOINT_RESPONSES,
        }
    )
    _OPENAI_RESPONSES_ALIASES = frozenset({"response", "responses"})

    _hub: HuggingfaceHub
    _logger: Logger
    _participant_id: UUID
    _stack: AsyncExitStack

    @dataclass(frozen=True, slots=True, kw_only=True)
    class _RuntimeEnvelopeSelection:
        container: ContainerProfileSelection
        readiness_timeout_seconds: int = 30

    class AgentRuntimeEnvelopeLoader(Protocol):
        """Load an agent through a trusted runtime envelope."""

        trusted_runtime_envelope_runner: bool

        async def load_agent_runtime_envelope(
            self,
            plan: ContainerNormalizedRuntimeEnvelopePlan,
            *,
            path: str,
            agent_id: UUID,
            disable_memory: bool,
            uri: str | None,
            tool_settings: ToolSettingsContext | None,
            event_manager_mode: EventManagerMode,
        ) -> Orchestrator:
            """Load an agent through a trusted runtime envelope."""

    @staticmethod
    def _is_trusted_runtime_envelope_loader(loader: object) -> bool:
        return (
            getattr(loader, "trusted_runtime_envelope_runner", False) is True
        )

    def __init__(
        self,
        *,
        hub: HuggingfaceHub,
        logger: Logger,
        participant_id: UUID,
        stack: AsyncExitStack,
        runtime_envelope_loader: AgentRuntimeEnvelopeLoader | None = None,
    ) -> None:
        self._hub = hub
        self._logger = logger
        self._participant_id = participant_id
        self._stack = stack
        if runtime_envelope_loader is not None:
            assert hasattr(
                runtime_envelope_loader,
                "load_agent_runtime_envelope",
            )
            assert self._is_trusted_runtime_envelope_loader(
                runtime_envelope_loader
            ), "runtime envelope loader must be trusted"
        self._runtime_envelope_loader = runtime_envelope_loader

    @staticmethod
    def _sentence_transformer_model_type() -> type[Any]:
        global SentenceTransformerModel
        if SentenceTransformerModel is None:
            module = import_module("avalan.model.nlp.sentence")
            SentenceTransformerModel = cast(
                type[Any], getattr(module, "SentenceTransformerModel")
            )
        return SentenceTransformerModel

    @staticmethod
    def _text_partitioner_type() -> type[Any]:
        global TextPartitioner
        if TextPartitioner is None:
            module = import_module("avalan.memory.partitioner.text")
            TextPartitioner = cast(
                type[Any], getattr(module, "TextPartitioner")
            )
        return TextPartitioner

    @staticmethod
    def _pgsql_raw_memory_type() -> type[Any]:
        global PgsqlRawMemory
        if PgsqlRawMemory is None:
            module = import_module("avalan.memory.permanent.pgsql.raw")
            PgsqlRawMemory = cast(type[Any], getattr(module, "PgsqlRawMemory"))
        return PgsqlRawMemory

    @staticmethod
    def _needs_text_partitioner(
        settings: OrchestratorSettings,
        tool_settings: ToolSettingsContext | None,
    ) -> bool:
        if settings.memory_permanent_message or settings.permanent_memory:
            return True

        browser_settings = tool_settings.browser if tool_settings else None
        return bool(browser_settings and browser_settings.search)

    def _load_text_partitioner(
        self,
        settings: OrchestratorSettings,
    ) -> Partitioner:
        _l = self._log_wrapper(self._logger)
        sentence_model_engine_settings = (
            TransformerEngineSettings(**settings.sentence_model_engine_config)
            if settings.sentence_model_engine_config
            else TransformerEngineSettings()
        )

        _l(
            "Loading sentence transformer model %s for agent %s",
            settings.sentence_model_id,
            settings.agent_id,
        )

        sentence_model_type = self._sentence_transformer_model_type()
        sentence_model_resource = sentence_model_type(
            model_id=settings.sentence_model_id,
            settings=sentence_model_engine_settings,
            logger=self._logger,
        )
        sentence_model = self._stack.enter_context(sentence_model_resource)

        _l(
            "Loading text partitioner for model %s for agent %s with settings"
            " (%s, %s, %s)",
            settings.sentence_model_id,
            settings.agent_id,
            settings.sentence_model_max_tokens,
            settings.sentence_model_overlap_size,
            settings.sentence_model_window_size,
        )

        text_partitioner_type = self._text_partitioner_type()
        return cast(
            Partitioner,
            text_partitioner_type(
                model=sentence_model,
                logger=self._logger,
                max_tokens=settings.sentence_model_max_tokens,
                overlap_size=settings.sentence_model_overlap_size,
                window_size=settings.sentence_model_window_size,
            ),
        )

    @staticmethod
    def parse_permanent_store_value(
        value: str,
    ) -> PermanentMemoryStoreSettings:
        raw_value = value.strip()
        description: str | None = None
        if "," in raw_value:
            dsn, description_part = raw_value.split(",", 1)
            description = description_part.strip() or None
        else:
            dsn = raw_value
        dsn = dsn.strip()
        assert dsn, "Permanent memory store DSN must be provided"
        return PermanentMemoryStoreSettings(dsn=dsn, description=description)

    @classmethod
    def _parse_serve_protocols(
        cls, raw_protocols: list[str] | None
    ) -> dict[str, set[str]] | None:
        if not raw_protocols:
            return None

        selection: dict[str, set[str]] = {}
        for raw_protocol in raw_protocols:
            assert raw_protocol, "Protocol value cannot be empty"
            protocol_part, _, endpoints_part = raw_protocol.partition(":")
            protocol = protocol_part.strip().lower()
            assert protocol, "Protocol name cannot be empty"
            assert (
                protocol in cls._ALLOWED_PROTOCOLS
            ), f"Unsupported protocol '{protocol}'"

            endpoints_text = endpoints_part.strip()
            if endpoints_text:
                assert (
                    protocol == "openai"
                ), "Only the openai protocol accepts endpoint selection"
                endpoints = selection.setdefault(protocol, set())
                for endpoint in endpoints_text.split(","):
                    endpoint_name = endpoint.strip().lower()
                    assert (
                        endpoint_name
                    ), "OpenAI endpoint name cannot be empty"
                    if endpoint_name in cls._OPENAI_COMPLETION_ALIASES:
                        endpoints.add(cls._OPENAI_ENDPOINT_COMPLETIONS)
                    elif endpoint_name in cls._OPENAI_RESPONSES_ALIASES:
                        endpoints.add(cls._OPENAI_ENDPOINT_RESPONSES)
                    else:
                        raise AssertionError(
                            f"Unsupported OpenAI endpoint '{endpoint_name}'"
                        )
            else:
                if protocol == "openai":
                    selection[protocol] = set(cls._OPENAI_ENDPOINTS)
                else:
                    selection[protocol] = set()

        return selection

    @classmethod
    async def _load_serve_protocol_strings(
        cls,
        path: str,
    ) -> list[str] | None:
        config = toml_loads(await read_text(path))

        serve_section = config.get("serve")
        if serve_section is None:
            return None

        assert isinstance(serve_section, dict), "Serve section must be a table"

        raw_protocols = serve_section.get("protocols")
        if raw_protocols is None:
            return None

        assert isinstance(
            raw_protocols, list
        ), "Serve protocols must be defined as a list"

        parsed_protocols: list[str] = []
        for item in raw_protocols:
            assert isinstance(
                item, str
            ), "Serve protocol entries must be strings"
            value = item.strip()
            assert value, "Serve protocol entries cannot be empty"
            parsed_protocols.append(value)

        return parsed_protocols

    @classmethod
    async def resolve_serve_protocols(
        cls,
        *,
        specs_path: str | None,
        cli_protocols: list[str] | None,
    ) -> dict[str, set[str]] | None:
        protocols = cls._parse_serve_protocols(cli_protocols)
        if protocols is not None:
            return protocols

        if not specs_path:
            return None

        config_protocols = await cls._load_serve_protocol_strings(specs_path)
        return cls._parse_serve_protocols(config_protocols)

    @property
    def hub(self) -> HuggingfaceHub:
        return self._hub

    @property
    def participant_id(self) -> UUID:
        return self._participant_id

    @classmethod
    async def validate_agent_file(cls, path: str) -> dict[str, Any]:
        if not exists(path):
            raise FileNotFoundError(path)
        if not access(path, R_OK):
            raise PermissionError(path)

        config = toml_loads(await read_text(path))

        config = await cls._resolve_agent_config_schema_refs(
            config,
            path=path,
        )
        cls.validate_agent_config(config)
        return config

    @classmethod
    def validate_agent_config(cls, config: object) -> None:
        assert isinstance(
            config, dict
        ), "Agent configuration must be a mapping"
        assert "agent" in config, "No agent section in configuration"
        assert "engine" in config, "No engine section defined in configuration"
        assert isinstance(
            config["agent"], dict
        ), "Agent section must be a mapping"
        assert isinstance(
            config["engine"], dict
        ), "Engine section must be a mapping"
        assert (
            "uri" in config["engine"]
        ), "No uri defined in engine section of configuration"
        cls._validate_engine_file_delivery_profile(config["engine"])
        cls._validate_agent_container_config(config)
        agent_config = config["agent"]
        cls._validate_agent_section(agent_config)
        orchestrator_type = agent_config.get("type")
        assert orchestrator_type is None or orchestrator_type in ["json"], (
            f"Unknown type {agent_config['type']} in agent section "
            + "of configuration"
        )

    @classmethod
    def _validate_agent_container_config(
        cls,
        config: Mapping[str, object],
    ) -> None:
        tool_section = config.get("tool", {})
        assert isinstance(tool_section, dict), "Tool section must be a mapping"
        source = trusted_container_source(ContainerSurface.AGENT_TOML)
        runtime_envelope = cls._validate_tool_isolation_policy(
            config,
            tool_section,
            source,
        )
        container_settings = cls._container_settings_from_tool_section(
            tool_section,
            source,
        )
        cls._container_runtime_settings_from_config(config, tool_section)
        cls._isolation_runtime_settings_from_config(config, tool_section)
        if runtime_envelope is not None:
            assert (
                container_settings is not None
            ), "runtime.container requires tool.container trusted settings"
            container_settings.select_profile(runtime_envelope.container)

    @classmethod
    def _container_runtime_settings_from_config(
        cls,
        config: Mapping[str, object],
        tool_section: Mapping[str, object],
    ) -> ContainerToolRuntimeSettings | None:
        source = trusted_container_source(ContainerSurface.AGENT_TOML)
        shell_selection = cls._shell_container_selection_from_tool_section(
            tool_section,
            source,
        )
        container_settings = cls._container_settings_from_tool_section(
            tool_section,
            source,
            shell_selection=shell_selection,
        )
        if container_settings is None or shell_selection is None:
            return None
        return ContainerToolRuntimeSettings(
            effective_settings=container_settings.select_profile(
                shell_selection
            ),
            rootful_authorized=source.can_define_runtime_authority,
        )

    @classmethod
    def _isolation_runtime_settings_from_config(
        cls,
        config: Mapping[str, object],
        tool_section: Mapping[str, object],
    ) -> IsolationToolRuntimeSettings | None:
        source = trusted_isolation_source(IsolationSettingsSurface.AGENT_TOML)
        shell_selection = cls._shell_sandbox_selection_from_tool_section(
            tool_section,
            source,
        )
        sandbox_settings = cls._sandbox_settings_from_tool_section(
            tool_section,
            source,
            shell_selection=shell_selection,
        )
        if sandbox_settings is None or shell_selection is None:
            return None
        settings = IsolationSettings(
            source=source,
            mode=IsolationMode.SANDBOX,
            sandbox=sandbox_settings,
        )
        return IsolationToolRuntimeSettings(
            effective_settings=settings.select_profile(
                IsolationProfileSelection(
                    mode=IsolationMode.SANDBOX,
                    profile=shell_selection.profile,
                    required=shell_selection.required,
                )
            ),
        )

    @classmethod
    def _validate_tool_isolation_policy(
        cls,
        config: Mapping[str, object],
        tool_section: Mapping[str, object],
        source: ContainerSettingsSource,
    ) -> _RuntimeEnvelopeSelection | None:
        runtime_envelope = cls._runtime_container_selection_from_config(
            config,
            source,
        )
        has_container = "container" in tool_section
        has_sandbox = "sandbox" in tool_section
        assert not (
            has_container and has_sandbox
        ), "tool cannot mix sandbox and container policy"
        shell_mode = cls._shell_execution_mode_from_tool_section(tool_section)
        if has_sandbox:
            assert (
                shell_mode == "sandbox"
            ), "tool.sandbox requires tool.shell backend sandbox"
        if has_container:
            assert (
                shell_mode == "container" or runtime_envelope is not None
            ), "tool.container requires tool.shell backend container"
        return runtime_envelope

    @staticmethod
    def _container_settings_from_tool_section(
        tool_section: Mapping[str, object],
        source: ContainerSettingsSource,
        *,
        shell_selection: ContainerProfileSelection | None = None,
    ) -> ContainerSettings | None:
        container_config = tool_section.get("container")
        if container_config is None:
            assert (
                shell_selection is None or shell_selection.profile is None
            ), "required container profile unavailable"
            return None
        assert isinstance(
            container_config,
            dict,
        ), "tool.container section must be a mapping"
        settings = trusted_container_settings_from_mapping(
            container_config,
            source=source,
        )
        assert settings.backend in {
            ContainerBackend.DOCKER,
            ContainerBackend.APPLE_CONTAINER,
        }, "tool.container backend must be docker or apple-container"
        return settings

    @staticmethod
    def _sandbox_settings_from_tool_section(
        tool_section: Mapping[str, object],
        source: IsolationSettingsSource,
        *,
        shell_selection: SandboxProfileSelection | None = None,
    ) -> SandboxSettings | None:
        sandbox_config = tool_section.get("sandbox")
        if sandbox_config is None:
            assert (
                shell_selection is None or shell_selection.profile is None
            ), "required sandbox profile unavailable"
            return None
        assert isinstance(
            sandbox_config,
            dict,
        ), "tool.sandbox section must be a mapping"
        settings = SandboxSettings.from_dict(sandbox_config, source=source)
        if shell_selection is not None:
            settings.select_profile(shell_selection)
        return settings

    @classmethod
    def _shell_container_selection_from_tool_section(
        cls,
        tool_section: Mapping[str, object],
        source: ContainerSettingsSource,
    ) -> ContainerProfileSelection | None:
        shell_config = tool_section.get("shell")
        if shell_config is None:
            return None
        assert isinstance(
            shell_config,
            dict,
        ), "tool.shell section must be a mapping"
        assert not (
            "container" in shell_config and "sandbox" in shell_config
        ), "tool.shell cannot mix sandbox and container policy"
        shell_mode = cls._shell_execution_mode_from_config(shell_config)
        shell_container_config = shell_config.get("container")
        if shell_container_config is not None:
            return cls._shell_container_selection_from_config(
                shell_config,
                shell_container_config,
                source,
            )
        if shell_mode == "container":
            return ContainerProfileSelection(required=True)
        return None

    @staticmethod
    def _shell_container_selection_from_config(
        shell_config: Mapping[str, object],
        shell_container_config: object,
        source: ContainerSettingsSource,
    ) -> ContainerProfileSelection:
        assert isinstance(
            shell_container_config,
            dict,
        ), "tool.shell.container section must be a mapping"
        assert isinstance(source, ContainerSettingsSource)
        assert (
            OrchestratorLoader._shell_execution_mode_from_config(shell_config)
            == "container"
        ), "tool.shell.container requires tool.shell backend container"
        selection_config = dict(shell_container_config)
        if "required" in selection_config:
            assert (
                selection_config["required"] is True
            ), "tool.shell backend container requires required=true"
        selection_config["required"] = True
        return container_selection_from_mapping(
            selection_config,
            source=source,
        )

    @classmethod
    def _shell_sandbox_selection_from_tool_section(
        cls,
        tool_section: Mapping[str, object],
        source: IsolationSettingsSource,
    ) -> SandboxProfileSelection | None:
        shell_config = tool_section.get("shell")
        if shell_config is None:
            return None
        assert isinstance(
            shell_config,
            dict,
        ), "tool.shell section must be a mapping"
        assert not (
            "container" in shell_config and "sandbox" in shell_config
        ), "tool.shell cannot mix sandbox and container policy"
        shell_mode = cls._shell_execution_mode_from_config(shell_config)
        shell_sandbox_config = shell_config.get("sandbox")
        if shell_sandbox_config is not None:
            return cls._shell_sandbox_selection_from_config(
                shell_config,
                shell_sandbox_config,
                source,
            )
        if shell_mode == "sandbox":
            return SandboxProfileSelection(required=True)
        return None

    @staticmethod
    def _shell_sandbox_selection_from_config(
        shell_config: Mapping[str, object],
        shell_sandbox_config: object,
        source: IsolationSettingsSource,
    ) -> SandboxProfileSelection:
        assert isinstance(
            shell_sandbox_config,
            dict,
        ), "tool.shell.sandbox section must be a mapping"
        assert isinstance(source, IsolationSettingsSource)
        assert (
            OrchestratorLoader._shell_execution_mode_from_config(shell_config)
            == "sandbox"
        ), "tool.shell.sandbox requires tool.shell backend sandbox"
        selection_config = dict(shell_sandbox_config)
        if "required" in selection_config:
            assert (
                selection_config["required"] is True
            ), "tool.shell backend sandbox requires required=true"
        selection_config["required"] = True
        return SandboxProfileSelection.from_dict(
            selection_config,
            source=source,
        )

    @classmethod
    def _shell_execution_mode_from_tool_section(
        cls,
        tool_section: Mapping[str, object],
    ) -> str | None:
        shell_config = tool_section.get("shell")
        if shell_config is None:
            return None
        assert isinstance(
            shell_config,
            dict,
        ), "tool.shell section must be a mapping"
        return cls._shell_execution_mode_from_config(shell_config)

    @staticmethod
    def _shell_execution_mode_from_config(
        shell_config: Mapping[str, object],
    ) -> str | None:
        backend = shell_config.get("backend")
        execution_mode = shell_config.get("execution_mode")
        if backend is not None:
            assert isinstance(
                backend,
                str,
            ), "tool.shell backend must be a string"
            assert backend in {
                "local",
                "sandbox",
                "container",
            }, "tool.shell backend must be local, sandbox, or container"
        if execution_mode is not None:
            assert isinstance(
                execution_mode,
                str,
            ), "tool.shell execution_mode must be a string"
            assert execution_mode in {
                "local",
                "sandbox",
                "container",
            }, "tool.shell execution_mode must be local, sandbox, or container"
        if backend is not None and execution_mode is not None:
            assert (
                backend == execution_mode
            ), "tool.shell backend and execution_mode must match"
        return backend or execution_mode

    @staticmethod
    def _runtime_container_selection_from_config(
        config: Mapping[str, object],
        source: ContainerSettingsSource,
    ) -> _RuntimeEnvelopeSelection | None:
        runtime_config = config.get("runtime")
        if runtime_config is None:
            return None
        assert isinstance(
            runtime_config,
            dict,
        ), "runtime section must be a mapping"
        container_config = runtime_config.get("container")
        if container_config is None:
            return None
        assert isinstance(
            container_config,
            dict,
        ), "runtime.container section must be a mapping"
        selection_config = dict(container_config)
        readiness_timeout_seconds = selection_config.pop(
            "readiness_timeout_seconds",
            30,
        )
        assert isinstance(readiness_timeout_seconds, int)
        assert not isinstance(readiness_timeout_seconds, bool)
        assert readiness_timeout_seconds > 0
        if "required" in selection_config:
            assert (
                selection_config["required"] is True
            ), "runtime.container requires required=true"
        selection_config["required"] = True
        return OrchestratorLoader._RuntimeEnvelopeSelection(
            container=container_selection_from_mapping(
                selection_config,
                source=source,
                scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
            ),
            readiness_timeout_seconds=readiness_timeout_seconds,
        )

    @staticmethod
    def _agent_runtime_envelope_plan(
        *,
        config: Mapping[str, object],
        tool_section: Mapping[str, object],
        source: ContainerSettingsSource,
        path: str,
        agent_id: UUID,
    ) -> ContainerNormalizedRuntimeEnvelopePlan | None:
        envelope = OrchestratorLoader._runtime_container_selection_from_config(
            config,
            source,
        )
        if envelope is None:
            return None
        container_settings = (
            OrchestratorLoader._container_settings_from_tool_section(
                tool_section,
                source,
            )
        )
        assert (
            container_settings is not None
        ), "runtime.container requires tool.container trusted settings"
        agent_section = config.get("agent", {})
        assert isinstance(agent_section, Mapping)
        logical_name = str(agent_section.get("name") or agent_id)
        effective = container_settings.select_profile(envelope.container)
        return normalize_runtime_envelope_plan(
            effective,
            ContainerPlanRequest(
                request_kind=ContainerPlanRequestKind.AGENT_SESSION,
                logical_name=logical_name,
                command="avalan-agent",
                argv=("avalan", "agent", "run", path),
                cwd="/workspace",
                scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                request_id=str(agent_id),
            ),
            envelope_kind=ContainerRuntimeEnvelopeKind.WHOLE_AGENT,
            readiness_timeout_seconds=envelope.readiness_timeout_seconds,
        )

    async def from_file(
        self,
        path: str,
        *,
        agent_id: UUID | None,
        disable_memory: bool = False,
        uri: str | None = None,
        tool_settings: ToolSettingsContext | None = None,
        tool_name_policy: ToolNamePolicySettings | None = None,
        event_manager_mode: EventManagerMode = EventManagerMode.SDK,
    ) -> Orchestrator:
        _l = self._log_wrapper(self._logger)
        assert isinstance(event_manager_mode, EventManagerMode)

        if not exists(path):
            raise FileNotFoundError(path)
        elif not access(path, R_OK):
            raise PermissionError(path)

        _l("Loading agent from %s", path, is_debug=False)

        config = toml_loads(await read_text(path))
        config = await OrchestratorLoader._resolve_agent_config_schema_refs(
            config,
            path=path,
        )
        # Validate settings

        assert "agent" in config, "No agent section in configuration"
        assert "engine" in config, "No engine section defined in configuration"
        assert (
            "uri" in config["engine"]
        ), "No uri defined in engine section of configuration"

        agent_config = config["agent"]
        self._validate_agent_section(agent_config)

        assert "engine" in config, "No engine section defined in configuration"
        assert (
            "uri" in config["engine"]
        ), "No uri defined in engine section of configuration"

        uri = uri or config["engine"]["uri"]
        engine_config = config["engine"]
        OrchestratorLoader._validate_engine_file_delivery_profile(
            engine_config
        )
        assert "tools" not in engine_config, (
            "tools option in [engine] is no longer supported; "
            "configure tools under [tool.enable]"
        )
        tool_section = config.get("tool")
        if tool_section is None:
            tool_section = {}
        else:
            assert isinstance(
                tool_section, dict
            ), "Tool section must be a mapping"
        skills_config = self._skills_config_from_tool_section(tool_section)
        container_source = trusted_container_source(
            ContainerSurface.AGENT_TOML
        )
        runtime_envelope = self._validate_tool_isolation_policy(
            config,
            tool_section,
            container_source,
        )
        self._container_settings_from_tool_section(
            tool_section,
            container_source,
        )
        container_runtime = self._container_runtime_settings_from_config(
            config,
            tool_section,
        )
        isolation_runtime = self._isolation_runtime_settings_from_config(
            config,
            tool_section,
        )

        enable_tools_config = tool_section.get("enable")
        enable_tools: list[str] | None = None
        if enable_tools_config is not None:
            if isinstance(enable_tools_config, str):
                enable_tools = [enable_tools_config]
            else:
                assert isinstance(
                    enable_tools_config, list
                ), "tool.enable must be a string or a list of strings"
                enable_tools = []
                for tool_name in enable_tools_config:
                    assert isinstance(
                        tool_name, str
                    ), "tool.enable entries must be strings"
                    enable_tools.append(tool_name)
        enable_tools = normalize_shell_enabled_tools(enable_tools)
        engine_config.pop("uri", None)
        engine_config.pop("file_delivery_profile", None)
        orchestrator_type = (
            config["agent"]["type"] if "type" in config["agent"] else None
        )
        agent_id = (
            agent_id
            if agent_id
            else (
                config["agent"]["id"] if "id" in config["agent"] else uuid4()
            )
        )

        assert orchestrator_type is None or orchestrator_type in ["json"], (
            f"Unknown type {config['agent']['type']} in agent section "
            + "of configuration"
        )

        call_options = config["run"] if "run" in config else None
        if call_options is not None:
            assert isinstance(
                call_options, dict
            ), "Run section must be a mapping"
        if call_options and "chat" in call_options:
            call_options["chat_settings"] = call_options.pop("chat")
        template_vars = config["template"] if "template" in config else None

        # Memory configuration

        memory_options = (
            config["memory"]
            if "memory" in config and not disable_memory
            else None
        )

        memory_permanent_message = (
            memory_options["permanent_message"]
            if memory_options and "permanent_message" in memory_options
            else None
        )

        memory_permanent: dict[str, PermanentMemoryStoreSettings] | None = None
        if memory_options and "permanent" in memory_options:
            memory_permanent_option = memory_options["permanent"]
            assert isinstance(
                memory_permanent_option, dict
            ), "Permanent memory should be a mapping"
            memory_permanent = {
                str(ns): OrchestratorLoader.parse_permanent_store_value(
                    str(dsn)
                )
                for ns, dsn in memory_permanent_option.items()
            }
        memory_recent = (
            memory_options["recent"]
            if memory_options and "recent" in memory_options
            else False
        )
        assert isinstance(
            memory_recent, bool
        ), "Recent message memory can only be set or unset"

        sentence_model_id = (
            config["memory.engine"]["model_id"]
            if "memory.engine" in config
            and "model_id" in config["memory.engine"]
            else OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID
        )
        sentence_model_engine_config = (
            config["memory.engine"] if "memory.engine" in config else None
        )
        sentence_model_max_tokens = (
            config["memory.engine"]["max_tokens"]
            if sentence_model_engine_config
            and "max_tokens" in sentence_model_engine_config
            else OrchestratorLoader.DEFAULT_SENTENCE_MODEL_MAX_TOKENS
        )
        sentence_model_overlap_size = (
            config["memory.engine"]["overlap_size"]
            if sentence_model_engine_config
            and "overlap_size" in sentence_model_engine_config
            else OrchestratorLoader.DEFAULT_SENTENCE_MODEL_OVERLAP_SIZE
        )
        sentence_model_window_size = (
            config["memory.engine"]["window_size"]
            if sentence_model_engine_config
            and "window_size" in sentence_model_engine_config
            else OrchestratorLoader.DEFAULT_SENTENCE_MODEL_WINDOW_SIZE
        )

        if sentence_model_engine_config:
            sentence_model_engine_config.pop("model_id", None)
            sentence_model_engine_config.pop("max_tokens", None)
            sentence_model_engine_config.pop("overlap_size", None)
            sentence_model_engine_config.pop("window_size", None)

        settings = OrchestratorSettings(
            agent_id=agent_id,
            orchestrator_type=orchestrator_type,
            agent_config=agent_config,
            uri=uri,
            engine_config=engine_config,
            tools=enable_tools,
            call_options=call_options,
            template_vars=template_vars,
            memory_permanent_message=memory_permanent_message,
            permanent_memory=memory_permanent,
            memory_recent=memory_recent,
            sentence_model_id=sentence_model_id,
            sentence_model_engine_config=sentence_model_engine_config,
            sentence_model_max_tokens=sentence_model_max_tokens,
            sentence_model_overlap_size=sentence_model_overlap_size,
            sentence_model_window_size=sentence_model_window_size,
            json_config=(
                config.get("json") if isinstance(config, dict) else None
            ),
            log_events=True,
        )

        browser_config = None
        browser_section = tool_section.get("browser")
        if browser_section is not None:
            assert isinstance(
                browser_section, dict
            ), "tool.browser section must be a mapping"
            browser_open_section = browser_section.get("open")
            if browser_open_section is not None:
                assert isinstance(
                    browser_open_section, dict
                ), "tool.browser.open section must be a mapping"
                browser_config = browser_open_section
            else:
                browser_config = browser_section
        browser_settings = None
        if browser_config:
            if "debug_source" in browser_config and isinstance(
                browser_config["debug_source"], str
            ):
                browser_config["debug_source"] = open(
                    browser_config["debug_source"]
                )
            browser_settings = BrowserToolSettings(**browser_config)

        database_settings = None
        database_config = tool_section.get("database")
        if database_config:
            assert isinstance(
                database_config, dict
            ), "tool.database section must be a mapping"
            database_settings = DatabaseToolSettings(**database_config)

        graph_settings = None
        graph_config = tool_section.get("graph")
        if graph_config:
            assert isinstance(
                graph_config, dict
            ), "tool.graph section must be a mapping"
            graph_settings = GraphToolSettings(**graph_config)

        shell_settings = None
        if "shell" in tool_section:
            shell_section = tool_section["shell"]
            assert isinstance(
                shell_section, dict
            ), "tool.shell section must be a mapping"
            shell_config = dict(shell_section)
            shell_container_config = shell_config.pop("container", None)
            shell_sandbox_config = shell_config.pop("sandbox", None)
            assert not (
                shell_container_config is not None
                and shell_sandbox_config is not None
            ), "tool.shell cannot mix sandbox and container policy"
            if shell_container_config is not None:
                shell_settings_source = trusted_container_source(
                    ContainerSurface.AGENT_TOML
                )
                shell_config["container"] = (
                    self._shell_container_selection_from_config(
                        shell_config,
                        shell_container_config,
                        shell_settings_source,
                    )
                )
            if shell_sandbox_config is not None:
                shell_sandbox_source = trusted_isolation_source(
                    IsolationSettingsSurface.AGENT_TOML
                )
                shell_config["sandbox"] = (
                    self._shell_sandbox_selection_from_config(
                        shell_config,
                        shell_sandbox_config,
                        shell_sandbox_source,
                    )
                )
            shell_settings = ShellToolSettings(**shell_config)

        skills_settings = None
        extra: dict[str, object] | None
        if tool_settings:
            browser_settings = tool_settings.browser or browser_settings
            database_settings = tool_settings.database or database_settings
            graph_settings = tool_settings.graph or graph_settings
            skills_settings = tool_settings.skills
            shell_settings = _merge_shell_tool_settings(
                shell_settings,
                tool_settings.shell,
                explicit_fields=tool_settings.shell_explicit_fields,
            )
            container_runtime = tool_settings.container or container_runtime
            isolation_runtime = tool_settings.isolation or isolation_runtime
            extra = tool_settings.extra
        else:
            extra = None

        if skills_config is not None:
            (
                skills_settings,
                skills_narrowing_config,
            ) = self._trusted_manifest_skills_settings_from_config(
                skills_config,
                trusted=skills_settings,
            )
            if skills_narrowing_config:
                assert (
                    skills_settings is not None
                ), "tool.skills requires trusted skills settings"
                skills_override = self._untrusted_skills_settings_from_config(
                    skills_narrowing_config,
                    trusted=skills_settings,
                )
                skills_merge = merge_skill_settings(
                    skills_settings,
                    skills_override,
                )
                assert not skills_merge.diagnostics, (
                    skills_merge.diagnostics[0].message
                    if skills_merge.diagnostics
                    else "Invalid tool.skills settings"
                )
                skills_settings = skills_merge.settings

        tool_settings = ToolSettingsContext(
            browser=browser_settings,
            database=database_settings,
            graph=graph_settings,
            skills=skills_settings,
            shell=shell_settings,
            shell_explicit_fields=(
                tool_settings.shell_explicit_fields if tool_settings else None
            ),
            container=container_runtime,
            isolation=isolation_runtime,
            extra=extra,
        )

        if runtime_envelope is not None:
            runtime_envelope_plan = self._agent_runtime_envelope_plan(
                config=config,
                tool_section=tool_section,
                source=container_source,
                path=path,
                agent_id=agent_id,
            )
            assert runtime_envelope_plan is not None
            assert (
                self._runtime_envelope_loader is not None
            ), "runtime.container requires an envelope-aware agent loader"
            return (
                await (
                    self._runtime_envelope_loader.load_agent_runtime_envelope(
                        runtime_envelope_plan,
                        path=path,
                        agent_id=agent_id,
                        disable_memory=disable_memory,
                        uri=uri,
                        tool_settings=tool_settings,
                        event_manager_mode=event_manager_mode,
                    )
                )
            )

        tool_format = None
        tool_format_str = tool_section.get("format")
        if tool_format_str:
            tool_format = ToolFormat(tool_format_str)
        parsed_tool_name_policy = self._tool_name_policy_from_tool_section(
            tool_section
        )
        tool_name_policy = tool_name_policy or parsed_tool_name_policy

        recovery_format_values = tool_section.get("recovery_formats", [])
        assert isinstance(
            recovery_format_values, list
        ), "tool.recovery_formats must be a list"
        tool_recovery_formats: list[ToolCallRecoveryFormat] = []
        for value in recovery_format_values:
            assert isinstance(
                value, str
            ), "tool.recovery_formats entries must be strings"
            tool_recovery_formats.append(ToolCallRecoveryFormat(value))

        _l("Loaded agent from %s", path, is_debug=False)

        if tool_recovery_formats:
            if event_manager_mode is not EventManagerMode.SDK:
                return await self.from_settings(
                    settings,
                    tool_settings=tool_settings,
                    tool_format=tool_format,
                    tool_name_policy=tool_name_policy,
                    tool_recovery_formats=tool_recovery_formats,
                    event_manager_mode=event_manager_mode,
                )
            return await self.from_settings(
                settings,
                tool_settings=tool_settings,
                tool_format=tool_format,
                tool_name_policy=tool_name_policy,
                tool_recovery_formats=tool_recovery_formats,
            )
        if event_manager_mode is not EventManagerMode.SDK:
            return await self.from_settings(
                settings,
                tool_settings=tool_settings,
                tool_format=tool_format,
                tool_name_policy=tool_name_policy,
                event_manager_mode=event_manager_mode,
            )
        return await self.from_settings(
            settings,
            tool_settings=tool_settings,
            tool_format=tool_format,
            tool_name_policy=tool_name_policy,
        )

    @classmethod
    def _skills_config_from_tool_section(
        cls,
        tool_section: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        skills_config = tool_section.get("skills")
        if skills_config is None:
            return None
        assert isinstance(
            skills_config,
            dict,
        ), "tool.skills section must be a mapping"
        return skills_config

    @classmethod
    def _trusted_manifest_skills_settings_from_config(
        cls,
        skills_config: Mapping[str, object],
        *,
        trusted: TrustedSkillSettings | None,
    ) -> tuple[TrustedSkillSettings | None, Mapping[str, object]]:
        manifest_sources = cls._manifest_skill_sources_from_config(
            skills_config.get("files"),
        )
        manifest_auto_enable = cls._manifest_auto_enable_from_config(
            skills_config,
            default=(
                trusted.manifest_auto_enable if trusted is not None else True
            ),
        )
        narrowing_config = {
            key: value
            for key, value in skills_config.items()
            if key
            not in {
                "files",
                "file_auto_enable",
                "manifest_auto_enable",
            }
        }
        if not manifest_sources:
            if trusted is None:
                return None, narrowing_config
            return (
                replace_trusted_skill_settings(
                    trusted,
                    manifest_auto_enable=manifest_auto_enable,
                ),
                narrowing_config,
            )

        if trusted is None:
            return (
                TrustedSkillSettings(
                    sources=manifest_sources,
                    manifest_auto_enable=manifest_auto_enable,
                ),
                narrowing_config,
            )

        duplicate_labels = {source.label for source in trusted.sources} & {
            source.label for source in manifest_sources
        }
        assert not duplicate_labels, "tool.skills.files labels must be unique"
        return (
            replace_trusted_skill_settings(
                trusted,
                sources=(*trusted.sources, *manifest_sources),
                manifest_auto_enable=manifest_auto_enable,
            ),
            narrowing_config,
        )

    @staticmethod
    def _manifest_auto_enable_from_config(
        skills_config: Mapping[str, object],
        *,
        default: bool,
    ) -> bool:
        assert isinstance(default, bool)
        has_manifest_key = "manifest_auto_enable" in skills_config
        has_file_key = "file_auto_enable" in skills_config
        assert not (
            has_manifest_key and has_file_key
        ), "tool.skills cannot mix manifest_auto_enable and file_auto_enable"
        if has_manifest_key:
            value = skills_config["manifest_auto_enable"]
        elif has_file_key:
            value = skills_config["file_auto_enable"]
        else:
            return default
        assert isinstance(
            value,
            bool,
        ), "tool.skills.manifest_auto_enable must be a boolean"
        return value

    @classmethod
    def _manifest_skill_sources_from_config(
        cls,
        files_config: object,
    ) -> tuple[SkillSourceConfig, ...]:
        if files_config is None:
            return ()
        assert isinstance(
            files_config,
            Mapping,
        ), "tool.skills.files must be a mapping"
        sources: list[SkillSourceConfig] = []
        for label, config in files_config.items():
            assert isinstance(
                label,
                str,
            ), "tool.skills.files labels must be strings"
            sources.append(
                cls._manifest_skill_source_from_config(label, config)
            )
        return tuple(sources)

    @staticmethod
    def _manifest_skill_source_from_config(
        label: str,
        config: object,
    ) -> SkillSourceConfig:
        assert label.strip(), "tool.skills.files label must be non-empty"
        if isinstance(config, str):
            return SkillSourceConfig(
                label=label,
                authority=WorkspaceSkillSourceAuthority(),
                manifest_path=config,
            )
        assert isinstance(
            config,
            Mapping,
        ), "tool.skills.files entries must be strings or mappings"
        unknown_keys = sorted(
            set(config) - {"allow_hidden", "authority", "path"}
        )
        assert not unknown_keys, "tool.skills.files entry has unknown keys"
        path = config.get("path")
        assert isinstance(
            path,
            str,
        ), "tool.skills.files entry path must be a string"
        authority_value = config.get("authority")
        authority: SkillSourceAuthority
        if authority_value is None:
            authority = WorkspaceSkillSourceAuthority()
        else:
            assert isinstance(
                authority_value,
                str,
            ), "tool.skills.files entry authority must be a string"
            authority = OrchestratorLoader._skill_source_authority_from_config(
                authority_value,
            )
        allow_hidden = config.get("allow_hidden", False)
        assert isinstance(
            allow_hidden,
            bool,
        ), "tool.skills.files entry allow_hidden must be a boolean"
        return SkillSourceConfig(
            label=label,
            authority=authority,
            manifest_path=path,
            allow_hidden_paths=allow_hidden,
        )

    @classmethod
    def _untrusted_skills_settings_from_config(
        cls,
        skills_config: Mapping[str, object],
        *,
        trusted: TrustedSkillSettings,
    ) -> UntrustedSkillSettings:
        return parse_untrusted_skill_settings_config(
            skills_config,
            trusted=trusted,
            surface=SkillSettingsSurface.AGENT,
            section="tool.skills",
        )

    @staticmethod
    def _skill_source_authority_from_config(
        value: str,
    ) -> SkillSourceAuthority:
        kind_value, separator, identity = value.partition(":")
        try:
            kind = SkillSourceAuthorityKind(kind_value)
        except ValueError as exc:
            raise AssertionError(
                "unsupported skills source authority"
            ) from exc
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
            assert (
                identity
            ), "plugin_provided skills authority requires plugin id"
            return PluginProvidedSkillSourceAuthority(plugin_id=identity)
        if kind is SkillSourceAuthorityKind.PREINSTALLED_REMOTE:
            assert (
                identity
            ), "preinstalled_remote skills authority requires registry id"
            return PreinstalledRemoteSkillSourceAuthority(registry_id=identity)
        raise AssertionError("unsupported skills source authority")

    @staticmethod
    def _tool_name_policy_from_tool_section(
        tool_section: dict[str, Any],
    ) -> ToolNamePolicySettings | None:
        policy_config = tool_section.get("name_policy")
        if policy_config is None:
            return None
        assert isinstance(
            policy_config, dict
        ), "tool.name_policy section must be a mapping"
        supported_keys = {
            "mode",
            "prefix",
            "replacement",
            "collapse_replacement",
            "provider_family",
            "map",
        }
        unknown_keys = sorted(set(policy_config) - supported_keys)
        assert not unknown_keys, "tool.name_policy has unknown keys"

        mode_value = policy_config.get(
            "mode", ToolNamePolicyMode.ENCODED.value
        )
        assert isinstance(
            mode_value, str
        ), "tool.name_policy.mode must be a string"

        prefix = policy_config.get("prefix", "avl_")
        assert isinstance(
            prefix, str
        ), "tool.name_policy.prefix must be a string"
        replacement = policy_config.get("replacement", "_")
        assert isinstance(
            replacement, str
        ), "tool.name_policy.replacement must be a string"
        collapse = policy_config.get("collapse_replacement", True)
        assert isinstance(
            collapse, bool
        ), "tool.name_policy.collapse_replacement must be a boolean"
        provider_family = policy_config.get("provider_family")
        assert provider_family is None or isinstance(
            provider_family, str
        ), "tool.name_policy.provider_family must be a string"
        name_map = policy_config.get("map", {})
        assert isinstance(
            name_map, dict
        ), "tool.name_policy.map must be a mapping"
        for canonical_name, provider_name in name_map.items():
            assert isinstance(
                canonical_name, str
            ), "tool.name_policy.map keys must be strings"
            assert isinstance(
                provider_name, str
            ), "tool.name_policy.map values must be strings"

        return ToolNamePolicySettings(
            mode=ToolNamePolicyMode(mode_value),
            prefix=prefix,
            replacement=replacement,
            collapse_replacement=collapse,
            provider_family=provider_family,
            map=dict(name_map),
        )

    async def from_settings(
        self,
        settings: OrchestratorSettings,
        *,
        tool_settings: ToolSettingsContext | None = None,
        tool_format: ToolFormat | None = None,
        tool_name_policy: ToolNamePolicySettings | None = None,
        tool_recovery_formats: list[ToolCallRecoveryFormat] | None = None,
        event_manager_mode: EventManagerMode = EventManagerMode.SDK,
    ) -> Orchestrator:
        _l = self._log_wrapper(self._logger)
        assert isinstance(event_manager_mode, EventManagerMode)

        _l("Loading agent from settings", is_debug=False)

        def load_text_partitioner() -> Partitioner:
            return self._load_text_partitioner(settings)

        text_partitioner = (
            load_text_partitioner()
            if self._needs_text_partitioner(settings, tool_settings)
            else None
        )

        _l("Loading event manager")

        event_manager = EventManager(mode=event_manager_mode)
        if settings.log_events:

            def _log_event(event: Event) -> None:
                is_info_event = event.type in (
                    EventType.TOOL_PROCESS,
                    EventType.TOOL_RESULT,
                )
                _l(
                    "%s",
                    event.payload,
                    inner_type=f"Event {event.type}",
                    is_debug=not is_info_event,
                )

            event_manager.add_listener(_log_event)

        _l("Loading memory manager for agent %s", settings.agent_id)

        memory = await MemoryManager.create_instance(
            agent_id=settings.agent_id,
            participant_id=self._participant_id,
            text_partitioner=text_partitioner,
            text_partitioner_factory=load_text_partitioner,
            logger=self._logger,
            with_permanent_message_memory=settings.memory_permanent_message,
            with_recent_message_memory=settings.memory_recent,
            event_manager=event_manager,
        )

        for namespace, store_settings in (
            settings.permanent_memory or {}
        ).items():
            _l(
                "Loading permanent memory %s for agent %s",
                namespace,
                settings.agent_id,
            )
            pgsql_raw_memory_type = self._pgsql_raw_memory_type()
            store = await pgsql_raw_memory_type.create_instance(
                dsn=store_settings.dsn, logger=self._logger
            )
            memory.add_permanent_memory(
                namespace,
                store,
                description=store_settings.description,
            )

        if text_partitioner:
            _l(
                "Loading tool manager for agent %s with partitioner and a"
                " sentence model %s with settings (%s, %s, %s)",
                settings.agent_id,
                settings.sentence_model_id,
                settings.sentence_model_max_tokens,
                settings.sentence_model_overlap_size,
                settings.sentence_model_window_size,
            )
        else:
            _l("Loading tool manager for agent %s", settings.agent_id)

        browser_settings = tool_settings.browser if tool_settings else None
        database_settings = tool_settings.database if tool_settings else None
        graph_settings = tool_settings.graph if tool_settings else None
        skills_settings = tool_settings.skills if tool_settings else None
        shell_settings = tool_settings.shell if tool_settings else None
        container_runtime = tool_settings.container if tool_settings else None
        isolation_runtime = tool_settings.isolation if tool_settings else None
        enabled_tools = normalize_shell_enabled_tools(settings.tools)
        if _skills_tools_requested(enabled_tools):
            assert (
                skills_settings is not None
            ), "skills tools require trusted skills settings"
            assert (
                skills_settings.enabled
            ), "skills tools require enabled trusted skills settings"
        effective_enabled_tools = effective_skills_enabled_tools(
            skills_settings,
            enabled_tools,
        )

        _l(
            "Tool settings: browser=%s, database=%s, graph=%s, shell=%s",
            browser_settings,
            database_settings,
            graph_settings,
            shell_settings,
        )

        available_toolsets = [
            MathToolSet(namespace="math"),
            MemoryToolSet(memory, namespace="memory"),
        ]
        if should_append_a2a_toolset(enabled_tools):
            available_toolsets.insert(1, A2AToolSet(namespace="a2a"))
        if should_append_mcp_toolset(enabled_tools):
            available_toolsets.insert(1, McpToolSet(namespace="mcp"))
        if HAS_GRAPH_DEPENDENCIES:
            available_toolsets.append(
                GraphToolSet(
                    settings=graph_settings or GraphToolSettings(),
                    namespace="graph",
                )
            )
        if HAS_CODE_DEPENDENCIES:
            available_toolsets.append(
                CodeToolSet(
                    container_runtime=container_runtime,
                    namespace="code",
                )
            )
        if HAS_BROWSER_DEPENDENCIES:
            available_toolsets.append(
                BrowserToolSet(
                    settings=browser_settings or BrowserToolSettings(),
                    partitioner=text_partitioner,
                    namespace="browser",
                )
            )
        if database_settings:
            available_toolsets.append(
                DatabaseToolSet(
                    settings=database_settings, namespace="database"
                )
            )
        if should_append_skills_toolset(skills_settings, enabled_tools):
            assert skills_settings is not None
            available_toolsets.append(
                await _build_skills_toolset(
                    skills_settings,
                    event_manager=event_manager,
                )
            )
        active_shell_settings = None
        if should_append_shell_toolset(
            shell_settings=shell_settings,
            enabled_tools=enabled_tools,
        ):
            candidate_shell_settings = shell_settings or ShellToolSettings()
            (
                shell_container_runtime,
                shell_isolation_runtime,
            ) = _shell_tool_runtime_settings(
                candidate_shell_settings,
                container_runtime,
                isolation_runtime,
            )
            shell_toolset_kwargs: dict[str, Any] = {
                "settings": candidate_shell_settings,
                "namespace": "shell",
            }
            if shell_isolation_runtime is not None:
                shell_toolset_kwargs["isolation_runtime"] = (
                    shell_isolation_runtime
                )
            else:
                shell_toolset_kwargs["container_runtime"] = (
                    shell_container_runtime
                )
            shell_toolset = ShellToolSet(**shell_toolset_kwargs)
            available_toolsets.append(shell_toolset)
            if _toolset_exposes_enabled_tool(
                shell_toolset,
                enabled_tools,
            ) or enables_shell_pipeline(
                enabled_tools,
                candidate_shell_settings,
            ):
                active_shell_settings = candidate_shell_settings

        tool = ToolManager.create_instance(
            available_toolsets=available_toolsets,
            enable_tools=effective_enabled_tools,
            settings=ToolManagerSettings(
                tool_format=tool_format,
                tool_name_policy=(
                    tool_name_policy or ToolNamePolicySettings()
                ),
                filters=(
                    [shell_input_file_filter(active_shell_settings)]
                    if active_shell_settings is not None
                    else None
                ),
                recovery_formats=tool_recovery_formats or [],
            ),
        )
        tool = await self._stack.enter_async_context(tool)

        _l(
            "Creating orchestrator %s #%s",
            settings.orchestrator_type,
            settings.agent_id,
        )

        model_manager = ModelManager(
            self._hub, self._logger, event_manager=event_manager
        )
        model_manager = self._stack.enter_context(model_manager)

        engine_uri = model_manager.parse_uri(settings.uri)
        engine_settings = model_manager.get_engine_settings(
            engine_uri,
            settings=settings.engine_config,
        )

        assert settings.agent_id

        if settings.orchestrator_type == "json":
            assert settings.json_config is not None
            agent: Orchestrator = self._load_json_orchestrator(
                agent_id=settings.agent_id,
                engine_uri=engine_uri,
                engine_settings=engine_settings,
                logger=self._logger,
                model_manager=model_manager,
                memory=memory,
                tool=tool,
                event_manager=event_manager,
                config={"json": settings.json_config},
                agent_config=settings.agent_config,
                call_options=settings.call_options,
                shell_input_file_settings=active_shell_settings,
                template_vars=settings.template_vars,
            )
        else:
            has_direct_prompt = (
                "system" in settings.agent_config
                or "developer" in settings.agent_config
            )
            agent = DefaultOrchestrator(
                engine_uri,
                self._logger,
                model_manager,
                memory,
                tool,
                event_manager,
                id=settings.agent_id,
                name=settings.agent_config.get("name"),
                role=(
                    None
                    if has_direct_prompt
                    else settings.agent_config.get("role")
                ),
                task=(
                    None
                    if has_direct_prompt
                    else settings.agent_config.get("task")
                ),
                goal_instructions=(
                    None
                    if has_direct_prompt
                    else settings.agent_config.get("goal_instructions")
                ),
                instructions=settings.agent_config.get("instructions"),
                rules=settings.agent_config.get("rules"),
                system=settings.agent_config.get("system"),
                developer=settings.agent_config.get("developer"),
                user=settings.agent_config.get("user"),
                user_template=settings.agent_config.get("user_template"),
                settings=engine_settings,
                shell_input_file_settings=active_shell_settings,
                call_options=settings.call_options,
                template_vars=settings.template_vars,
            )

        _l("Loaded agent from settings", is_debug=False)

        return agent

    @staticmethod
    def _load_json_orchestrator(
        agent_id: UUID,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
        model_manager: ModelManager,
        memory: MemoryManager,
        tool: ToolManager,
        event_manager: EventManager,
        config: dict[str, Any],
        agent_config: dict[str, Any],
        call_options: dict[str, Any] | None,
        template_vars: dict[str, Any] | None,
        shell_input_file_settings: ShellToolSettings | None = None,
    ) -> JsonOrchestrator:
        assert "json" in config, "No json section in configuration"
        if (
            "system" not in agent_config
            and "developer" not in agent_config
            and "goal_instructions" in agent_config
        ):
            assert (
                "task" in agent_config
            ), "agent.goal_instructions requires agent.task"

        properties: list[Property] = []
        for property_name in config.get("json", []):
            output_property = config["json"][property_name]
            properties.append(
                Property(
                    name=property_name,
                    data_type=output_property["type"],
                    description=output_property["description"],
                )
            )

        assert properties, "No properties defined in configuration"

        has_direct_prompt = (
            "system" in agent_config or "developer" in agent_config
        )
        agent = JsonOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            properties,
            id=agent_id,
            name=agent_config["name"] if "name" in agent_config else None,
            role=(None if has_direct_prompt else agent_config.get("role")),
            task=(None if has_direct_prompt else agent_config.get("task")),
            goal_instructions=(
                None
                if has_direct_prompt
                else agent_config.get("goal_instructions")
            ),
            instructions=agent_config.get("instructions"),
            rules=agent_config.get("rules"),
            system=agent_config.get("system"),
            developer=agent_config.get("developer"),
            user=agent_config.get("user"),
            user_template=agent_config.get("user_template"),
            settings=engine_settings,
            shell_input_file_settings=shell_input_file_settings,
            call_options=call_options,
            template_vars=template_vars,
        )
        return agent

    @staticmethod
    def _validate_engine_file_delivery_profile(
        engine_config: dict[str, Any],
    ) -> None:
        value = engine_config.get("file_delivery_profile")
        if value is None:
            return
        assert isinstance(
            value, str
        ), "engine.file_delivery_profile must be a string"
        uri = engine_config.get("uri")
        assert isinstance(uri, str), "engine.uri must be a string"
        engine_uri = ModelManager.parse_uri(uri)
        assert (
            engine_uri.vendor is None
        ), "engine.file_delivery_profile is only supported for local models"
        assert value in {
            profile.value for profile in LocalFileDeliveryProfile
        }, "engine.file_delivery_profile is not supported"

    @staticmethod
    async def _resolve_agent_config_schema_refs(
        config: object,
        *,
        path: str,
    ) -> dict[str, Any]:
        assert isinstance(
            config, dict
        ), "Agent configuration must be a mapping"
        run_config = config.get("run")
        if run_config is None:
            return config
        assert isinstance(run_config, dict), "Run section must be a mapping"
        response_format = run_config.get("response_format")
        if response_format is None:
            return config
        assert isinstance(
            response_format, dict
        ), "run.response_format must be a mapping"
        run_config = dict(run_config)
        run_config["response_format"] = (
            await OrchestratorLoader._resolved_response_format(
                response_format,
                schema_base_path=path,
            )
        )
        config = dict(config)
        config["run"] = run_config
        return config

    @staticmethod
    async def _resolved_response_format(
        response_format: dict[str, Any],
        *,
        schema_base_path: str,
    ) -> dict[str, Any]:
        resolved = dict(response_format)
        await OrchestratorLoader._resolve_schema_ref_field(
            resolved,
            schema_base_path=schema_base_path,
            path="run.response_format.schema_ref",
        )
        nested = resolved.get("json_schema")
        if nested is not None:
            assert isinstance(
                nested, dict
            ), "run.response_format.json_schema must be a mapping"
            nested = dict(nested)
            await OrchestratorLoader._resolve_schema_ref_field(
                nested,
                schema_base_path=schema_base_path,
                path="run.response_format.json_schema.schema_ref",
            )
            resolved["json_schema"] = nested
        return resolved

    @staticmethod
    async def _resolve_schema_ref_field(
        mapping: dict[str, Any],
        *,
        schema_base_path: str,
        path: str,
    ) -> None:
        schema_ref = mapping.get("schema_ref")
        if schema_ref is None:
            return
        assert (
            "schema" not in mapping
        ), f"{path} cannot be used with an inline schema"
        try:
            schema = (
                await resolve_schema_ref(
                    schema_ref,
                    schema_base_path=schema_base_path,
                    path=path,
                )
            ).schema
        except TaskSchemaResolutionError as error:
            raise AssertionError(str(error)) from error
        mapping["schema"] = schema
        mapping.pop("schema_ref", None)

    @staticmethod
    def _validate_agent_section(agent_config: dict[str, Any]) -> None:
        assert not (
            "user" in agent_config and "user_template" in agent_config
        ), "user and user_template are mutually exclusive"
        assert not (
            "task" in agent_config
            and "instructions" in agent_config
            and "goal_instructions" not in agent_config
            and "system" not in agent_config
            and "developer" not in agent_config
        ), (
            "agent.instructions is reserved for provider instructions; "
            "use agent.goal_instructions with agent.task for goal "
            "instructions"
        )
        assert not (
            "goal_instructions" in agent_config and "task" not in agent_config
        ), "agent.goal_instructions requires agent.task"
        for key in (
            "system",
            "developer",
            "instructions",
            "goal_instructions",
            "user",
            "user_template",
        ):
            value = agent_config.get(key)
            assert value is None or isinstance(
                value, str
            ), f"agent.{key} must be a string"

    @staticmethod
    def _log_wrapper(logger: Logger) -> Callable[..., Any]:
        def wrapper(
            message: str,
            *args: Any,
            inner_type: str | None = None,
            **kwargs: Any,
        ) -> Any:
            is_debug = kwargs.pop("is_debug", True)
            level = DEBUG if is_debug else INFO
            prefix = (
                f"<{inner_type} @ OrchestratorLoader> "
                if inner_type
                else "<OrchestratorLoader> "
            )
            return logger.log(level, prefix + message, *args, **kwargs)

        return wrapper
