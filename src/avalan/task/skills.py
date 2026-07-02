from ..filesystem import read_text
from ..skill import (
    CANONICAL_SKILLS_TOOL_NAMES,
    SKILL_SETTINGS_POLICY_VERSION,
    SkillConfiguredSource,
    SkillDiagnosticInfo,
    SkillModelValue,
    SkillRegistry,
    SkillSettingsSurface,
    SkillSourceConfig,
    SkillStatus,
    TrustedSkillSettings,
    UntrustedSkillSettings,
    build_skill_registry,
    merge_skill_settings,
    parse_untrusted_skill_settings_config,
    resolve_skill_sources,
    trusted_skill_settings_fingerprint,
    trusted_skill_settings_identity_dict,
    trusted_skill_source_fingerprint,
    trusted_skill_source_identity_dict,
    untrusted_skill_settings_config_dict,
)
from ..skill.observability import (
    SkillEventPublisher,
    assert_skill_event_publisher,
    skill_audit_correlation_id,
)
from ..tool.names import matches_tool_namespace
from .definition import (
    TaskDefinition,
    TaskTargetType,
)
from .event import sanitize_raw_task_event_closed
from .observability import (
    ObservabilitySink,
    TaskObservedEvent,
    TaskSanitizedEventObserver,
)
from .privacy import PrivacySanitizer
from .schema import task_definition_schema_base_path
from .validation import (
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
)

from asyncio import gather
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from inspect import isawaitable
from json import dumps
from pathlib import Path, PurePosixPath, PureWindowsPath
from tomllib import TOMLDecodeError
from tomllib import loads as toml_loads

TASK_SKILLS_METADATA_KEY = "skills"
TASK_SKILLS_METADATA_VERSION = "task.skills.v1"
TASK_SKILLS_TARGETS = frozenset(
    {
        TaskTargetType.AGENT,
        TaskTargetType.FLOW,
        TaskTargetType.MODEL,
        TaskTargetType.TASK,
        TaskTargetType.TOOL,
    }
)

_CANONICAL_JSON_SEPARATORS = (",", ":")
_UNAVAILABLE_STATUSES = frozenset(
    {
        SkillStatus.EMPTY,
        SkillStatus.NOT_FOUND,
        SkillStatus.UNAVAILABLE,
    }
)


TaskRawEventObserver = Callable[..., Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class _TaskTomlDocument:
    raw: Mapping[str, object]
    source_path: Path
    root_path: Path


@dataclass(frozen=True, slots=True)
class TaskSkillAuditEventPublisher:
    sanitizer: PrivacySanitizer
    event_observer: TaskSanitizedEventObserver | None = None
    metrics_event_observer: TaskSanitizedEventObserver | None = None
    trace_event_observer: TaskSanitizedEventObserver | None = None
    observability_sink: ObservabilitySink | None = None
    raw_event_observer: TaskRawEventObserver | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.sanitizer, PrivacySanitizer)
        if self.event_observer is not None:
            assert callable(self.event_observer)
        if self.metrics_event_observer is not None:
            assert callable(self.metrics_event_observer)
        if self.trace_event_observer is not None:
            assert callable(self.trace_event_observer)
        if self.observability_sink is not None:
            assert callable(
                getattr(self.observability_sink, "record_event", None)
            )
        if self.raw_event_observer is not None:
            assert callable(self.raw_event_observer)

    async def trigger(self, event: object) -> None:
        if self.raw_event_observer is not None:
            result = self.raw_event_observer(event)
            await _await_task_skill_result(result)
            return
        observed = sanitize_raw_task_event_closed(event, self.sanitizer)
        await gather(
            _notify_task_skill_observer(self.event_observer, observed),
            _notify_task_skill_observer(
                self.metrics_event_observer,
                observed,
            ),
            _notify_task_skill_observer(
                self.trace_event_observer,
                observed,
            ),
            _record_task_skill_observability_event(
                self.observability_sink,
                observed,
            ),
        )


def task_skill_audit_event_publisher(
    *,
    sanitizer: PrivacySanitizer,
    event_observer: TaskSanitizedEventObserver | None = None,
    metrics_event_observer: TaskSanitizedEventObserver | None = None,
    trace_event_observer: TaskSanitizedEventObserver | None = None,
    observability_sink: ObservabilitySink | None = None,
    raw_event_observer: TaskRawEventObserver | None = None,
) -> TaskSkillAuditEventPublisher | None:
    assert isinstance(sanitizer, PrivacySanitizer)
    if event_observer is not None:
        assert callable(event_observer)
    if metrics_event_observer is not None:
        assert callable(metrics_event_observer)
    if trace_event_observer is not None:
        assert callable(trace_event_observer)
    if observability_sink is not None:
        assert callable(getattr(observability_sink, "record_event", None))
    if raw_event_observer is not None:
        assert callable(raw_event_observer)
    if (
        event_observer is None
        and metrics_event_observer is None
        and trace_event_observer is None
        and observability_sink is None
        and raw_event_observer is None
    ):
        # No configured audit delivery endpoint means there is nothing for
        # fail-closed semantics to require on this task surface.
        return None
    return TaskSkillAuditEventPublisher(
        sanitizer=sanitizer,
        event_observer=event_observer,
        metrics_event_observer=metrics_event_observer,
        trace_event_observer=trace_event_observer,
        observability_sink=observability_sink,
        raw_event_observer=raw_event_observer,
    )


async def _notify_task_skill_observer(
    observer: TaskSanitizedEventObserver | None,
    event: TaskObservedEvent,
) -> None:
    if observer is None:
        return
    result = observer(event)
    await _await_task_skill_result(result)


async def _await_task_skill_result(result: object) -> None:
    if isawaitable(result):
        await gather(result)


async def _record_task_skill_observability_event(
    sink: ObservabilitySink | None,
    event: TaskObservedEvent,
) -> None:
    if sink is None:
        return
    await sink.record_event(event)


async def task_definition_with_skills_identity(
    definition: TaskDefinition,
    *,
    trusted_settings: TrustedSkillSettings | None = None,
    registry: SkillRegistry | None = None,
    enabled_tools: Iterable[str] = (),
    event_manager: SkillEventPublisher | None = None,
    trust_definition_skills: bool = True,
    schema_base_path: str | Path | None = None,
) -> TaskDefinition:
    assert isinstance(definition, TaskDefinition)
    assert_skill_event_publisher(event_manager)
    settings = task_effective_skills_settings(
        definition,
        trusted_settings=trusted_settings,
        registry=registry,
        trust_definition_skills=trust_definition_skills,
    )
    if settings is None:
        if definition.skills_identity is not None:
            raise TaskValidationError((_task_skills_missing_issue(),))
        return definition
    if registry is None and settings.enabled:
        registry = await build_task_skill_registry(
            settings,
            event_manager=event_manager,
        )
    tools = tuple(enabled_tools) or await task_enabled_skills_tools(
        definition,
        schema_base_path=schema_base_path,
    )
    identity = task_skills_identity(
        settings,
        registry=registry,
        enabled_tools=tools,
        target_type=definition.execution.type,
    )
    return replace(
        definition,
        skills=settings,
        skills_identity=identity,
    )


async def revalidate_task_skills_for_worker(
    definition: TaskDefinition,
    *,
    trusted_settings: TrustedSkillSettings | None = None,
    registry: SkillRegistry | None = None,
    expected_identity: Mapping[str, object] | None = None,
    event_manager: SkillEventPublisher | None = None,
    schema_base_path: str | Path | None = None,
) -> TaskDefinition:
    assert isinstance(definition, TaskDefinition)
    assert_skill_event_publisher(event_manager)
    expected = (
        expected_identity
        if expected_identity is not None
        else definition.skills_identity
    )
    if expected is None:
        if (
            definition.skills is not None
            or definition.skills_config is not None
        ):
            raise TaskValidationError((_task_skills_missing_issue(),))
        return definition
    if not isinstance(expected, Mapping):
        raise TaskValidationError((_task_skills_stale_issue(),))
    settings = task_effective_skills_settings(
        definition,
        trusted_settings=trusted_settings,
        registry=registry,
        trust_definition_skills=False,
    )
    if settings is None:
        raise TaskValidationError((_task_skills_missing_issue(),))
    if registry is None and settings.enabled:
        registry = await build_task_skill_registry(
            settings,
            event_manager=event_manager,
        )
    tools = await task_enabled_skills_tools(
        definition,
        schema_base_path=schema_base_path,
        require_trusted_refs=True,
    )
    actual = task_skills_identity(
        settings,
        registry=registry,
        enabled_tools=tools,
        target_type=definition.execution.type,
    )
    issue = _task_skills_identity_issue(expected, actual)
    if issue is not None:
        raise TaskValidationError((issue,))
    return replace(definition, skills=settings, skills_identity=actual)


def task_effective_skills_settings(
    definition: TaskDefinition,
    *,
    trusted_settings: TrustedSkillSettings | None = None,
    registry: SkillRegistry | None = None,
    trust_definition_skills: bool = True,
) -> TrustedSkillSettings | None:
    assert isinstance(definition, TaskDefinition)
    if registry is not None:
        assert isinstance(registry, SkillRegistry)
    if definition.execution.type not in TASK_SKILLS_TARGETS:
        if (
            definition.skills is not None
            or definition.skills_config is not None
            or definition.skills_identity is not None
        ):
            raise TaskValidationError((_task_skills_unsupported_issue(),))
        return None
    if trust_definition_skills and definition.skills is not None:
        return definition.skills
    inherited = trusted_settings
    if inherited is None and registry is not None:
        inherited = registry.settings
    if inherited is not None:
        assert isinstance(inherited, TrustedSkillSettings)
    if definition.skills_config is None:
        return inherited
    if inherited is None:
        raise TaskValidationError((_task_skills_missing_issue(),))
    override = _rebase_task_skills_config(
        definition.skills_config,
        trusted=inherited,
    )
    result = merge_skill_settings(inherited, override)
    if result.diagnostics:
        raise TaskValidationError(
            (_task_skills_policy_issue(result.diagnostics[0]),)
        )
    return result.settings


async def build_task_skill_registry(
    settings: TrustedSkillSettings,
    *,
    event_manager: SkillEventPublisher | None = None,
    audit_operation_id: str | None = None,
) -> SkillRegistry:
    assert isinstance(settings, TrustedSkillSettings)
    assert_skill_event_publisher(event_manager)
    assert audit_operation_id is None or isinstance(audit_operation_id, str)
    if event_manager is not None and audit_operation_id is None:
        audit_operation_id = skill_audit_correlation_id(
            "task-skill-registry-build"
        )
    sources = _skill_configured_sources(settings)
    resolution = await resolve_skill_sources(
        sources,
        settings=settings,
        event_manager=event_manager,
        audit_operation_id=audit_operation_id,
    )
    return await build_skill_registry(
        resolution,
        settings=settings,
        event_manager=event_manager,
        audit_operation_id=audit_operation_id,
    )


def task_skills_identity(
    settings: TrustedSkillSettings,
    *,
    registry: SkillRegistry | None,
    enabled_tools: Iterable[str],
    target_type: TaskTargetType,
) -> dict[str, object]:
    assert isinstance(settings, TrustedSkillSettings)
    assert isinstance(target_type, TaskTargetType)
    if registry is not None:
        assert isinstance(registry, SkillRegistry)
    registry_version = (
        registry.registry_version.as_model_value()
        if registry is not None
        else None
    )
    settings_identity = trusted_skill_settings_identity_dict(settings)
    registry_identity = registry_version or _fingerprint(settings_identity)
    status = (
        registry.status.value
        if registry is not None
        else (
            SkillStatus.DISABLED.value
            if not settings.enabled
            else SkillStatus.UNAVAILABLE.value
        )
    )
    source_labels = (
        tuple(source.label for source in settings.sources)
        if settings.privacy.include_source_labels
        else ()
    )
    value: dict[str, object] = {
        "version": TASK_SKILLS_METADATA_VERSION,
        "target": target_type.value,
        "status": status,
        "settings_fingerprint": trusted_skill_settings_fingerprint(settings),
        "source_fingerprint": trusted_skill_source_fingerprint(settings),
        "policy_version": SKILL_SETTINGS_POLICY_VERSION,
        "enabled": settings.enabled,
        "bootstrap_enabled": settings.bootstrap_enabled,
        "read_limits": settings.read_limits.as_model_dict(),
        "index_limits": settings.index_limits.as_model_dict(),
        "source_limits": settings.source_limits.as_model_dict(),
        "cursor_limits": settings.cursor_limits.as_model_dict(),
        "privacy": settings.privacy.as_model_dict(),
        "observability": settings.observability.as_model_dict(),
        "registry_identity": registry_identity,
        "enabled_tools": tuple(_enabled_skills_tools(enabled_tools)),
        "authority_kinds": tuple(
            authority.value for authority in settings.authority_kinds
        ),
        "source_labels": source_labels,
        "allowed_skill_ids": settings.allowed_skill_ids,
    }
    if registry_version is not None:
        value["registry_version"] = registry_version
    return value


async def task_enabled_skills_tools(
    definition: TaskDefinition,
    *,
    schema_base_path: str | Path | None = None,
    require_trusted_refs: bool = False,
) -> tuple[str, ...]:
    assert isinstance(definition, TaskDefinition)
    assert isinstance(require_trusted_refs, bool)
    if definition.execution.type not in {
        TaskTargetType.AGENT,
        TaskTargetType.FLOW,
    }:
        return ()
    document = await _task_ref_toml(
        definition,
        schema_base_path=schema_base_path,
        require_trusted_refs=require_trusted_refs,
    )
    if document is None:
        return ()
    if definition.execution.type == TaskTargetType.FLOW:
        return await _flow_enabled_skills_tools(
            document,
            require_trusted_refs=require_trusted_refs,
        )
    return _agent_enabled_skills_tools(document.raw)


async def _task_ref_toml(
    definition: TaskDefinition,
    *,
    schema_base_path: str | Path | None,
    require_trusted_refs: bool,
) -> _TaskTomlDocument | None:
    base_path = task_definition_schema_base_path(
        definition,
        schema_base_path=schema_base_path,
    )
    if base_path is None:
        if require_trusted_refs:
            raise TaskValidationError((_task_skills_unavailable_issue(),))
        return None
    root = _task_ref_root(base_path)
    if root is None:
        if require_trusted_refs:
            raise TaskValidationError((_task_skills_unavailable_issue(),))
        return None
    return await _trusted_ref_toml(
        definition.execution.ref,
        root_path=root,
        require_trusted_refs=require_trusted_refs,
    )


async def _trusted_ref_toml(
    ref: object,
    *,
    root_path: Path,
    require_trusted_refs: bool,
) -> _TaskTomlDocument | None:
    if not isinstance(ref, str) or not ref.strip() or _is_untrusted_ref(ref):
        if require_trusted_refs:
            raise TaskValidationError((_task_skills_unavailable_issue(),))
        return None
    root = root_path.resolve(strict=False)
    ref_path = Path(ref)
    try:
        source_path = (root / ref_path).resolve(strict=False)
        source_path.relative_to(root)
        raw = toml_loads(await read_text(source_path))
    except TOMLDecodeError as error:
        if require_trusted_refs:
            raise TaskValidationError(
                (_task_skills_malformed_issue(),)
            ) from error
        return None
    except (OSError, RuntimeError, ValueError) as error:
        if require_trusted_refs:
            raise TaskValidationError(
                (_task_skills_unavailable_issue(),)
            ) from error
        return None
    return _TaskTomlDocument(raw=raw, source_path=source_path, root_path=root)


def _task_ref_root(base_path: str | Path) -> Path | None:
    if not isinstance(base_path, str | Path):
        return None
    base = Path(base_path)
    root = base.parent if base.suffix else base
    return root.resolve(strict=False)


def _agent_enabled_skills_tools(raw: Mapping[str, object]) -> tuple[str, ...]:
    tool = raw.get("tool")
    if not isinstance(tool, Mapping):
        return ()
    enabled = tool.get("enable")
    if isinstance(enabled, str):
        return tuple(_enabled_skills_tools((enabled,)))
    if not isinstance(enabled, list | tuple):
        return ()
    return tuple(
        _enabled_skills_tools(
            item for item in enabled if isinstance(item, str)
        )
    )


async def _flow_enabled_skills_tools(
    document: _TaskTomlDocument,
    *,
    require_trusted_refs: bool,
) -> tuple[str, ...]:
    nodes = document.raw.get("nodes")
    if not isinstance(nodes, Mapping):
        return ()
    names: list[str] = []
    for node in nodes.values():
        if not isinstance(node, Mapping):
            continue
        for key in ("ref", "type"):
            value = node.get(key)
            if isinstance(value, str):
                names.append(value)
        config = node.get("config")
        if isinstance(config, Mapping):
            for key in ("canonical_name", "tool", "name"):
                value = config.get(key)
                if isinstance(value, str):
                    names.append(value)
        if _is_agent_flow_node(node):
            agent_ref = node.get("ref")
            agent_document = await _trusted_ref_toml(
                agent_ref,
                root_path=document.root_path,
                require_trusted_refs=require_trusted_refs,
            )
            if agent_document is not None:
                names.extend(_agent_enabled_skills_tools(agent_document.raw))
    return tuple(_enabled_skills_tools(names))


def _is_agent_flow_node(node: Mapping[str, object]) -> bool:
    node_type = node.get("type")
    return isinstance(node_type, str) and node_type == "agent"


def _is_untrusted_ref(ref: str) -> bool:
    if "://" in ref or "\\" in ref:
        return True
    posix_path = PurePosixPath(ref)
    windows_path = PureWindowsPath(ref)
    if posix_path.is_absolute() or windows_path.is_absolute():
        return True
    return ".." in posix_path.parts or ".." in windows_path.parts


def task_skill_settings_allow(
    parent: TrustedSkillSettings,
    child: TrustedSkillSettings,
) -> bool:
    assert isinstance(parent, TrustedSkillSettings)
    assert isinstance(child, TrustedSkillSettings)
    if child.enabled and not parent.enabled:
        return False
    if child.bootstrap_enabled and not parent.bootstrap_enabled:
        return False
    if not set(child.authority_kinds).issubset(set(parent.authority_kinds)):
        return False
    if not _sources_allow(parent, child):
        return False
    if parent.allowed_skill_ids:
        if not child.allowed_skill_ids:
            return False
        if not set(child.allowed_skill_ids).issubset(
            set(parent.allowed_skill_ids)
        ):
            return False
    return (
        parent.read_limits.allows(child.read_limits)
        and parent.index_limits.allows(child.index_limits)
        and parent.source_limits.allows(child.source_limits)
        and parent.cursor_limits.allows(child.cursor_limits)
        and parent.privacy.allows(child.privacy)
        and parent.observability.allows(child.observability)
    )


def _rebase_task_skills_config(
    override: UntrustedSkillSettings,
    *,
    trusted: TrustedSkillSettings,
) -> UntrustedSkillSettings:
    assert isinstance(override, UntrustedSkillSettings)
    assert isinstance(trusted, TrustedSkillSettings)
    return parse_untrusted_skill_settings_config(
        untrusted_skill_settings_config_dict(override),
        trusted=trusted,
        surface=SkillSettingsSurface.TASK,
        section="skills",
    )


def _task_skills_identity_issue(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> TaskValidationIssue | None:
    expected = _identity_mapping(expected)
    actual = _identity_mapping(actual)
    expected_status = expected.get("status")
    actual_status = actual.get("status")
    if actual_status == SkillStatus.POLICY_DENIED.value:
        return _task_skills_policy_denied_issue()
    if actual_status == SkillStatus.MALFORMED.value:
        return _task_skills_malformed_issue()
    if actual_status in {status.value for status in _UNAVAILABLE_STATUSES}:
        return _task_skills_unavailable_issue()
    if expected_status == SkillStatus.POLICY_DENIED.value:
        return _task_skills_policy_denied_issue()
    if expected_status == SkillStatus.MALFORMED.value:
        return _task_skills_malformed_issue()
    if _identity_widened(expected, actual):
        return _task_skills_widened_issue()
    if dict(expected) != dict(actual):
        return _task_skills_stale_issue()
    return None


def _identity_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    normalized = _identity_value(value)
    assert isinstance(normalized, Mapping)
    return normalized


def _identity_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _identity_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return tuple(_identity_value(item) for item in value)
    return value


def _identity_widened(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> bool:
    if actual.get("enabled") is True and expected.get("enabled") is False:
        return True
    if actual.get("bootstrap_enabled") is True and (
        expected.get("bootstrap_enabled") is False
    ):
        return True
    expected_tools = set(_string_items(expected.get("enabled_tools")))
    actual_tools = set(_string_items(actual.get("enabled_tools")))
    if actual_tools > expected_tools:
        return True
    expected_sources = set(_string_items(expected.get("source_labels")))
    actual_sources = set(_string_items(actual.get("source_labels")))
    if expected_sources and actual_sources > expected_sources:
        return True
    expected_skills = set(_string_items(expected.get("allowed_skill_ids")))
    actual_skills = set(_string_items(actual.get("allowed_skill_ids")))
    if expected_skills and (
        not actual_skills or actual_skills > expected_skills
    ):
        return True
    return _limits_widened(
        expected.get("read_limits"),
        actual.get("read_limits"),
    )


def _limits_widened(expected: object, actual: object) -> bool:
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        return False
    for key in ("max_bytes_per_read", "max_lines_per_read"):
        expected_value = expected.get(key)
        actual_value = actual.get(key)
        if (
            isinstance(expected_value, int)
            and not isinstance(expected_value, bool)
            and isinstance(actual_value, int)
            and not isinstance(actual_value, bool)
            and actual_value > expected_value
        ):
            return True
    return False


def _skill_configured_sources(
    settings: TrustedSkillSettings,
) -> tuple[SkillConfiguredSource, ...]:
    assert isinstance(settings, TrustedSkillSettings)
    sources: list[SkillConfiguredSource] = []
    for source in settings.sources:
        assert isinstance(source, SkillSourceConfig)
        if source.root_path is None:
            continue
        sources.append(
            SkillConfiguredSource(
                label=source.label,
                authority=source.authority,
                root_path=source.root_path,
                package_path=source.package_path,
                enabled=source.enabled,
                allow_hidden_paths=source.allow_hidden_paths,
            )
        )
    return tuple(sources)


def _sources_allow(
    parent: TrustedSkillSettings,
    child: TrustedSkillSettings,
) -> bool:
    if not child.sources_explicit or not child.sources:
        return True
    parent_sources = {
        source.label: trusted_skill_source_identity_dict(source)
        for source in parent.sources
    }
    for source in child.sources:
        if parent_sources.get(source.label) != (
            trusted_skill_source_identity_dict(source)
        ):
            return False
    return True


def _enabled_skills_tools(enabled_tools: Iterable[str]) -> tuple[str, ...]:
    requested: set[str] = set()
    for enabled in enabled_tools:
        if not isinstance(enabled, str) or not enabled.strip():
            continue
        for tool_name in CANONICAL_SKILLS_TOOL_NAMES:
            if matches_tool_namespace(tool_name, enabled):
                requested.add(tool_name)
    return tuple(sorted(requested))


def _string_items(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _fingerprint(value: Mapping[str, SkillModelValue]) -> str:
    payload = dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=_CANONICAL_JSON_SEPARATORS,
        sort_keys=True,
    )
    return f"task-skills:{sha256(payload.encode('utf-8')).hexdigest()}"


def _task_skills_unsupported_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="task.skills_unsupported_target",
        path="skills",
        message="Task skills settings are not supported for this target.",
        hint="Use skills only with agent, flow, model, tool, or task targets.",
        category=TaskValidationCategory.UNSUPPORTED,
    )


def _task_skills_missing_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="task.skills_registry_missing",
        path="skills",
        message="Queued task skills registry identity is missing.",
        hint="Run the task with trusted worker skills settings.",
        category=TaskValidationCategory.UNSUPPORTED,
    )


def _task_skills_unavailable_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="task.skills_registry_unavailable",
        path="skills",
        message="Queued task skills registry is unavailable.",
        hint="Restore the trusted skills sources before executing the task.",
        category=TaskValidationCategory.UNSUPPORTED,
    )


def _task_skills_malformed_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="task.skills_registry_malformed",
        path="skills",
        message="Queued task skills registry is malformed.",
        hint="Fix the trusted skills registry before executing the task.",
        category=TaskValidationCategory.VALUE,
    )


def _task_skills_widened_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="task.skills_registry_widened",
        path="skills",
        message="Queued task skills registry is wider than requested.",
        hint="Use the same or narrower trusted skills registry.",
        category=TaskValidationCategory.UNSUPPORTED,
    )


def _task_skills_stale_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="task.skills_registry_stale",
        path="skills",
        message="Queued task skills registry identity is stale.",
        hint="Re-enqueue the task after trusted skills policy changes.",
        category=TaskValidationCategory.VALUE,
    )


def _task_skills_policy_denied_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="task.skills_registry_policy_denied",
        path="skills",
        message="Queued task skills registry is policy denied.",
        hint="Use an authorized trusted skills registry.",
        category=TaskValidationCategory.UNSUPPORTED,
    )


def _task_skills_policy_issue(
    diagnostic: SkillDiagnosticInfo,
) -> TaskValidationIssue:
    assert isinstance(diagnostic, SkillDiagnosticInfo)
    return TaskValidationIssue(
        code="task.skills_registry_policy_denied",
        path="skills",
        message=diagnostic.message,
        hint=diagnostic.hint,
        category=TaskValidationCategory.UNSUPPORTED,
    )
