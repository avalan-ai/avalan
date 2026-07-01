from ..container import (
    ContainerDiagnostic,
    ContainerSurface,
    container_syntax_diagnostics,
)
from ..filesystem import (
    DEFAULT_TEXT_ENCODING,
    assert_text_encoding,
    read_text,
)
from ..skill import (
    SkillDiagnosticInfo,
    SkillSettingsSurface,
    TrustedSkillSettings,
    UntrustedSkillSettings,
    merge_skill_settings,
    parse_untrusted_skill_settings_config,
)
from .definition import (
    FrozenMetadata,
    IdempotencyMode,
    ObservabilitySinkType,
    PrivacyAction,
    RetryBackoff,
    RunMode,
    TaskArtifactPolicy,
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskInputType,
    TaskLimitsPolicy,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskOutputContract,
    TaskOutputType,
    TaskPrivacyPolicy,
    TaskRetryPolicy,
    TaskRunPolicy,
    TaskTargetType,
)
from .schema import (
    TaskSchemaResolutionError,
    resolve_task_definition_schemas,
)

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from tomllib import TOMLDecodeError, loads
from typing import TypeVar, cast

EnumValue = TypeVar("EnumValue", bound=StrEnum)
DefinitionValue = TypeVar("DefinitionValue")
RawSection = Mapping[str, object]

_REQUIRED_SECTIONS = ("task", "input", "output", "execution")
_SKILLS_TARGETS = frozenset(
    {
        TaskTargetType.AGENT,
        TaskTargetType.FLOW,
        TaskTargetType.MODEL,
        TaskTargetType.TASK,
        TaskTargetType.TOOL,
    }
)
_UNSUPPORTED_ISOLATION_PATHS = (
    "isolation",
    "sandbox",
    "sandboxProfile",
    "sandboxPolicy",
    "sandbox_profile",
    "sandbox_policy",
    "execution.isolation",
    "execution.sandbox",
    "execution.sandboxProfile",
    "execution.sandboxPolicy",
    "execution.sandbox_profile",
    "execution.sandbox_policy",
)


class TaskLoadIssueCategory(StrEnum):
    PARSE = "parse"
    STRUCTURE = "structure"
    UNSUPPORTED = "unsupported"
    VALUE = "value"


class TaskLoadSeverity(StrEnum):
    ERROR = "error"


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskLoadIssue:
    code: str
    path: str
    message: str
    hint: str
    category: TaskLoadIssueCategory
    severity: TaskLoadSeverity = TaskLoadSeverity.ERROR

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "path": self.path,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "hint": self.hint,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskLoadResult:
    definition: TaskDefinition | None
    issues: tuple[TaskLoadIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues and self.definition is not None


class TaskLoadError(ValueError):
    issues: tuple[TaskLoadIssue, ...]

    def __init__(self, issues: tuple[TaskLoadIssue, ...]) -> None:
        assert issues, "issues must not be empty"
        self.issues = issues
        summary = ", ".join(
            f"{issue.code} at {issue.path}" for issue in issues
        )
        super().__init__(f"task definition could not be loaded: {summary}")


class TaskDefinitionLoader:
    def __init__(
        self,
        *,
        encoding: str = DEFAULT_TEXT_ENCODING,
        skills_settings: TrustedSkillSettings | None = None,
    ) -> None:
        assert_text_encoding(encoding)
        if skills_settings is not None:
            assert isinstance(skills_settings, TrustedSkillSettings)
        self._encoding = encoding
        self._skills_settings = skills_settings

    async def load(
        self,
        path: str | Path,
        *,
        encoding: str | None = None,
        skills_settings: TrustedSkillSettings | None = None,
    ) -> TaskDefinition:
        result = await self.load_result(
            path,
            encoding=encoding,
            skills_settings=skills_settings,
        )
        if result.definition is None:
            raise TaskLoadError(result.issues)
        return result.definition

    async def load_result(
        self,
        path: str | Path,
        *,
        encoding: str | None = None,
        skills_settings: TrustedSkillSettings | None = None,
    ) -> TaskLoadResult:
        source_path = Path(path)
        source = await read_text(
            source_path,
            encoding=self._text_encoding(encoding),
        )
        return await self.loads_result(
            source,
            source_path=source_path,
            skills_settings=skills_settings,
        )

    def _text_encoding(self, encoding: str | None) -> str:
        if encoding is None:
            return self._encoding
        assert_text_encoding(encoding)
        return encoding

    async def loads(
        self,
        source: str,
        *,
        source_path: str | Path | None = None,
        skills_settings: TrustedSkillSettings | None = None,
    ) -> TaskDefinition:
        result = await self.loads_result(
            source,
            source_path=source_path,
            skills_settings=skills_settings,
        )
        if result.definition is None:
            raise TaskLoadError(result.issues)
        return result.definition

    async def loads_result(
        self,
        source: str,
        *,
        source_path: str | Path | None = None,
        skills_settings: TrustedSkillSettings | None = None,
    ) -> TaskLoadResult:
        assert isinstance(source, str), "source must be a string"
        try:
            raw = loads(source)
        except TOMLDecodeError:
            return TaskLoadResult(
                definition=None,
                issues=(
                    _issue(
                        code="task.malformed_toml",
                        path=_source_path(source_path),
                        message="Task definition TOML is malformed.",
                        hint="Fix the TOML syntax and retry loading.",
                        category=TaskLoadIssueCategory.PARSE,
                    ),
                ),
            )

        return await _build_definition(
            raw,
            source_path=source_path,
            trusted_skills=skills_settings or self._skills_settings,
        )


async def load_task_definition(
    path: str | Path,
    *,
    encoding: str = DEFAULT_TEXT_ENCODING,
    skills_settings: TrustedSkillSettings | None = None,
) -> TaskDefinition:
    return await TaskDefinitionLoader(encoding=encoding).load(
        path,
        skills_settings=skills_settings,
    )


async def load_task_definition_result(
    path: str | Path,
    *,
    encoding: str = DEFAULT_TEXT_ENCODING,
    skills_settings: TrustedSkillSettings | None = None,
) -> TaskLoadResult:
    return await TaskDefinitionLoader(encoding=encoding).load_result(
        path,
        skills_settings=skills_settings,
    )


async def loads_task_definition(
    source: str,
    *,
    source_path: str | Path | None = None,
    skills_settings: TrustedSkillSettings | None = None,
) -> TaskDefinition:
    return await TaskDefinitionLoader().loads(
        source,
        source_path=source_path,
        skills_settings=skills_settings,
    )


async def loads_task_definition_result(
    source: str,
    *,
    source_path: str | Path | None = None,
    skills_settings: TrustedSkillSettings | None = None,
) -> TaskLoadResult:
    return await TaskDefinitionLoader().loads_result(
        source,
        source_path=source_path,
        skills_settings=skills_settings,
    )


async def _build_definition(
    raw: Mapping[str, object],
    *,
    source_path: str | Path | None,
    trusted_skills: TrustedSkillSettings | None,
) -> TaskLoadResult:
    issues: list[TaskLoadIssue] = []
    issues.extend(_container_issues(raw))
    if issues:
        return TaskLoadResult(definition=None, issues=tuple(issues))
    sections = _sections(raw, issues)

    for section in _REQUIRED_SECTIONS:
        if section not in sections:
            issues.append(_missing_section(section))

    if issues:
        return TaskLoadResult(definition=None, issues=tuple(issues))

    task = _task_metadata(sections["task"], issues)
    input_contract = _input_contract(sections["input"], issues)
    output_contract = _output_contract(sections["output"], issues)
    execution = _execution_target(sections["execution"], issues)
    skills, skills_config = _skills_settings(
        sections.get("skills"),
        trusted_skills,
        execution_type=execution.type if execution is not None else None,
        issues=issues,
    )
    run = _run_policy(sections.get("run", {}), issues)
    retry = _retry_policy(sections.get("retry", {}), issues)
    privacy = _privacy_policy(sections.get("privacy", {}), issues)
    artifact = _artifact_policy(sections.get("artifact", {}), issues)
    limits = _limits_policy(sections.get("limits", {}), issues)
    observability = _observability_policy(
        sections.get("observability", {}), issues
    )

    if issues:
        return TaskLoadResult(definition=None, issues=tuple(issues))

    assert task is not None
    assert input_contract is not None
    assert output_contract is not None
    assert execution is not None
    assert run is not None
    assert retry is not None
    assert privacy is not None
    assert artifact is not None
    assert limits is not None
    assert observability is not None
    definition = TaskDefinition(
        task=task,
        input=input_contract,
        output=output_contract,
        execution=execution,
        definition_base=source_path,
        skills=skills,
        skills_config=skills_config,
        run=run,
        retry=retry,
        privacy=privacy,
        artifact=artifact,
        limits=limits,
        observability=observability,
    )
    try:
        definition = await resolve_task_definition_schemas(
            definition,
            schema_base_path=source_path,
        )
    except TaskSchemaResolutionError as error:
        issues.append(_schema_resolution_issue(error))
        return TaskLoadResult(definition=None, issues=tuple(issues))
    if (skills is not None or skills_config is not None) and source_path:
        definition = replace(definition, definition_base=source_path)
    return TaskLoadResult(definition=definition)


def _container_issues(
    raw: Mapping[str, object],
) -> tuple[TaskLoadIssue, ...]:
    return tuple(
        _issue_from_isolation_path(path)
        for path in _UNSUPPORTED_ISOLATION_PATHS
        if _has_path(raw, path)
    ) + tuple(
        _issue_from_container_diagnostic(diagnostic)
        for diagnostic in container_syntax_diagnostics(
            ContainerSurface.TASK_TOML,
            raw,
        )
    )


def _issue_from_container_diagnostic(
    diagnostic: ContainerDiagnostic,
) -> TaskLoadIssue:
    return _issue(
        code=diagnostic.code.value,
        path=diagnostic.path,
        message=diagnostic.message,
        hint=diagnostic.hint,
        category=TaskLoadIssueCategory.UNSUPPORTED,
    )


def _issue_from_isolation_path(path: str) -> TaskLoadIssue:
    return _issue(
        code="isolation.unsupported_syntax",
        path=path,
        message="Isolation syntax is not supported for task TOML yet.",
        hint=(
            f"Remove {path}; task isolation policy must be supplied by a "
            "trusted runtime or deployment."
        ),
        category=TaskLoadIssueCategory.UNSUPPORTED,
    )


def _has_path(raw: Mapping[str, object], dotted_path: str) -> bool:
    value: object = raw
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return False
        value = value[part]
    return True


def _sections(
    raw: Mapping[str, object], issues: list[TaskLoadIssue]
) -> dict[str, RawSection]:
    sections: dict[str, RawSection] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            issues.append(
                _issue(
                    code="task.invalid_section",
                    path=key,
                    message="Task definition sections must be tables.",
                    hint="Use TOML table syntax for task definition sections.",
                    category=TaskLoadIssueCategory.STRUCTURE,
                )
            )
        elif isinstance(key, str):
            sections[key] = value
    return sections


def _task_metadata(
    raw: RawSection, issues: list[TaskLoadIssue]
) -> TaskMetadata | None:
    issue_count = len(issues)
    name = _required_str(raw, "task.name", "name", issues)
    version = _required_str(raw, "task.version", "version", issues)
    description = _optional_str(raw, "task.description", "description", issues)
    labels = _string_tuple(raw, "task.labels", "labels", issues)
    annotations = _optional_mapping(
        raw, "task.annotations", "annotations", issues
    )
    if len(issues) > issue_count or name is None or version is None:
        return None
    return _construct(
        lambda: TaskMetadata(
            name=name,
            version=version,
            description=description,
            labels=labels,
            annotations=annotations or {},
        ),
        "task.invalid_value",
        "task",
        issues,
    )


def _input_contract(
    raw: RawSection, issues: list[TaskLoadIssue]
) -> TaskInputContract | None:
    issue_count = len(issues)
    input_type = _required_enum(
        raw, "input.type", "type", TaskInputType, issues
    )
    schema = _optional_mapping(raw, "input.schema", "schema", issues)
    schema_ref = _optional_str(raw, "input.schema_ref", "schema_ref", issues)
    description = _optional_str(
        raw, "input.description", "description", issues
    )
    required = _optional_bool(raw, "input.required", "required", issues, True)
    conversions = _string_tuple(
        raw, "input.file_conversions", "file_conversions", issues
    )
    mime_types = _string_tuple(raw, "input.mime_types", "mime_types", issues)
    if len(issues) > issue_count or input_type is None or required is None:
        return None
    return _construct(
        lambda: TaskInputContract(
            type=input_type,
            schema=schema,
            schema_ref=schema_ref,
            description=description,
            required=required,
            file_conversions=conversions,
            mime_types=mime_types,
        ),
        "input.invalid_value",
        "input",
        issues,
    )


def _output_contract(
    raw: RawSection, issues: list[TaskLoadIssue]
) -> TaskOutputContract | None:
    issue_count = len(issues)
    output_type = _required_enum(
        raw, "output.type", "type", TaskOutputType, issues
    )
    schema = _optional_mapping(raw, "output.schema", "schema", issues)
    schema_ref = _optional_str(raw, "output.schema_ref", "schema_ref", issues)
    description = _optional_str(
        raw, "output.description", "description", issues
    )
    if len(issues) > issue_count or output_type is None:
        return None
    return _construct(
        lambda: TaskOutputContract(
            type=output_type,
            schema=schema,
            schema_ref=schema_ref,
            description=description,
        ),
        "output.invalid_value",
        "output",
        issues,
    )


def _execution_target(
    raw: RawSection, issues: list[TaskLoadIssue]
) -> TaskExecutionTarget | None:
    issue_count = len(issues)
    target_type = _required_enum(
        raw, "execution.type", "type", TaskTargetType, issues
    )
    ref = _required_str(raw, "execution.ref", "ref", issues)
    variables = _optional_mapping(
        raw, "execution.variables", "variables", issues
    )
    if len(issues) > issue_count or target_type is None or ref is None:
        return None
    return _construct(
        lambda: TaskExecutionTarget(
            type=target_type,
            ref=ref,
            variables=variables or {},
        ),
        "execution.invalid_value",
        "execution",
        issues,
    )


def _skills_settings(
    raw: RawSection | None,
    trusted: TrustedSkillSettings | None,
    *,
    execution_type: TaskTargetType | None,
    issues: list[TaskLoadIssue],
) -> tuple[TrustedSkillSettings | None, UntrustedSkillSettings | None]:
    if trusted is not None:
        assert isinstance(trusted, TrustedSkillSettings)
    if raw is None:
        if execution_type in _SKILLS_TARGETS:
            return trusted, None
        return None, None
    if not isinstance(raw, Mapping):
        issues.append(_invalid_type("skills", "Use a TOML table value."))
        return None, None
    if execution_type not in _SKILLS_TARGETS:
        issues.append(
            _issue(
                code="task.skills_unsupported_target",
                path="skills",
                message="Task skills settings are not supported here.",
                hint=(
                    "Use skills settings only with agent, flow, model, tool, "
                    "or task execution targets."
                ),
                category=TaskLoadIssueCategory.UNSUPPORTED,
            )
        )
        return None, None
    if trusted is None:
        issues.append(
            _issue(
                code="task.skills_trusted_settings_required",
                path="skills",
                message="Task skills settings require trusted defaults.",
                hint=(
                    "Provide trusted skills settings from SDK, CLI, worker, "
                    "or operator configuration."
                ),
                category=TaskLoadIssueCategory.UNSUPPORTED,
            )
        )
        return None, None
    try:
        override = parse_untrusted_skill_settings_config(
            raw,
            trusted=trusted,
            surface=SkillSettingsSurface.TASK,
            section="skills",
        )
    except (AssertionError, ValueError) as error:
        issues.append(
            _issue(
                code="task.invalid_skills_settings",
                path="skills",
                message="Task skills settings are invalid.",
                hint=_assertion_hint(error),
                category=TaskLoadIssueCategory.VALUE,
            )
        )
        return None, None
    result = merge_skill_settings(trusted, override)
    issues.extend(
        _skill_issue(diagnostic) for diagnostic in result.diagnostics
    )
    if result.diagnostics:
        return None, override
    return result.settings, override


def _run_policy(
    raw: RawSection, issues: list[TaskLoadIssue]
) -> TaskRunPolicy | None:
    issue_count = len(issues)
    mode = _optional_enum(raw, "run.mode", "mode", RunMode, issues)
    timeout_seconds = _optional_int(
        raw, "run.timeout_seconds", "timeout_seconds", issues
    )
    idempotency = _optional_enum(
        raw, "run.idempotency", "idempotency", IdempotencyMode, issues
    )
    queue = _optional_str(raw, "run.queue", "queue", issues)
    priority = _optional_int(raw, "run.priority", "priority", issues)
    concurrency = _optional_int(raw, "run.concurrency", "concurrency", issues)
    idempotency_key_path = _optional_str(
        raw, "run.idempotency_key_path", "idempotency_key_path", issues
    )
    if len(issues) > issue_count:
        return None
    return _construct(
        lambda: TaskRunPolicy(
            mode=mode or RunMode.DIRECT,
            timeout_seconds=(
                timeout_seconds if timeout_seconds is not None else 300
            ),
            idempotency=idempotency or IdempotencyMode.NONE,
            queue=queue,
            priority=priority,
            concurrency=concurrency,
            idempotency_key_path=idempotency_key_path,
        ),
        "run.invalid_value",
        "run",
        issues,
    )


def _retry_policy(
    raw: RawSection, issues: list[TaskLoadIssue]
) -> TaskRetryPolicy | None:
    issue_count = len(issues)
    max_attempts = _optional_int(
        raw, "retry.max_attempts", "max_attempts", issues
    )
    backoff = _optional_enum(
        raw, "retry.backoff", "backoff", RetryBackoff, issues
    )
    max_delay_seconds = _optional_int(
        raw, "retry.max_delay_seconds", "max_delay_seconds", issues
    )
    jitter = _optional_bool(raw, "retry.jitter", "jitter", issues, False)
    if len(issues) > issue_count:
        return None
    return _construct(
        lambda: TaskRetryPolicy(
            max_attempts=max_attempts if max_attempts is not None else 1,
            backoff=backoff or RetryBackoff.NONE,
            max_delay_seconds=max_delay_seconds,
            jitter=bool(jitter),
        ),
        "retry.invalid_value",
        "retry",
        issues,
    )


def _privacy_policy(
    raw: RawSection, issues: list[TaskLoadIssue]
) -> TaskPrivacyPolicy | None:
    issue_count = len(issues)
    input_action = _optional_enum(
        raw, "privacy.input", "input", PrivacyAction, issues
    )
    prompt = _optional_enum(
        raw, "privacy.prompt", "prompt", PrivacyAction, issues
    )
    output = _optional_enum(
        raw, "privacy.output", "output", PrivacyAction, issues
    )
    files = _optional_enum(
        raw, "privacy.files", "files", PrivacyAction, issues
    )
    file_bytes = _optional_enum(
        raw, "privacy.file_bytes", "file_bytes", PrivacyAction, issues
    )
    token_text = _optional_enum(
        raw, "privacy.token_text", "token_text", PrivacyAction, issues
    )
    tool_arguments = _optional_enum(
        raw, "privacy.tool_arguments", "tool_arguments", PrivacyAction, issues
    )
    tool_results = _optional_enum(
        raw, "privacy.tool_results", "tool_results", PrivacyAction, issues
    )
    events = _optional_enum(
        raw, "privacy.events", "events", PrivacyAction, issues
    )
    errors = _optional_enum(
        raw, "privacy.errors", "errors", PrivacyAction, issues
    )
    raw_retention_days = _optional_int(
        raw, "privacy.raw_retention_days", "raw_retention_days", issues
    )
    if len(issues) > issue_count:
        return None
    return _construct(
        lambda: TaskPrivacyPolicy(
            input=input_action or PrivacyAction.HASH,
            prompt=prompt or PrivacyAction.REDACT,
            output=output or PrivacyAction.REDACT,
            files=files or PrivacyAction.HASH,
            file_bytes=file_bytes or PrivacyAction.DROP,
            token_text=token_text or PrivacyAction.DROP,
            tool_arguments=tool_arguments or PrivacyAction.REDACT,
            tool_results=tool_results or PrivacyAction.REDACT,
            events=events or PrivacyAction.REDACT,
            errors=errors or PrivacyAction.REDACT,
            raw_retention_days=raw_retention_days or 0,
        ),
        "privacy.invalid_value",
        "privacy",
        issues,
    )


def _artifact_policy(
    raw: RawSection, issues: list[TaskLoadIssue]
) -> TaskArtifactPolicy | None:
    issue_count = len(issues)
    retention_days = _optional_int(
        raw, "artifact.retention_days", "retention_days", issues
    )
    store_bytes = _optional_bool(
        raw, "artifact.store_bytes", "store_bytes", issues, False
    )
    storage = _optional_str(raw, "artifact.storage", "storage", issues)
    max_count = _optional_int(raw, "artifact.max_count", "max_count", issues)
    max_bytes = _optional_int(raw, "artifact.max_bytes", "max_bytes", issues)
    encrypt = _optional_bool(raw, "artifact.encrypt", "encrypt", issues, True)
    if len(issues) > issue_count:
        return None
    return _construct(
        lambda: TaskArtifactPolicy(
            retention_days=retention_days,
            store_bytes=bool(store_bytes),
            storage=storage,
            max_count=max_count,
            max_bytes=max_bytes,
            encrypt=bool(encrypt),
        ),
        "artifact.invalid_value",
        "artifact",
        issues,
    )


def _limits_policy(
    raw: RawSection, issues: list[TaskLoadIssue]
) -> TaskLimitsPolicy | None:
    issue_count = len(issues)
    input_bytes = _optional_int(
        raw, "limits.input_bytes", "input_bytes", issues
    )
    file_count = _optional_int(raw, "limits.file_count", "file_count", issues)
    file_bytes = _optional_int(raw, "limits.file_bytes", "file_bytes", issues)
    output_bytes = _optional_int(
        raw, "limits.output_bytes", "output_bytes", issues
    )
    artifact_count = _optional_int(
        raw, "limits.artifact_count", "artifact_count", issues
    )
    artifact_bytes = _optional_int(
        raw, "limits.artifact_bytes", "artifact_bytes", issues
    )
    total_tokens = _optional_int(
        raw, "limits.total_tokens", "total_tokens", issues
    )
    if len(issues) > issue_count:
        return None
    return _construct(
        lambda: TaskLimitsPolicy(
            input_bytes=input_bytes,
            file_count=file_count,
            file_bytes=file_bytes,
            output_bytes=output_bytes,
            artifact_count=artifact_count,
            artifact_bytes=artifact_bytes,
            total_tokens=total_tokens,
        ),
        "limits.invalid_value",
        "limits",
        issues,
    )


def _observability_policy(
    raw: RawSection, issues: list[TaskLoadIssue]
) -> TaskObservabilityPolicy | None:
    issue_count = len(issues)
    sinks = _enum_tuple(
        raw,
        "observability.sinks",
        "sinks",
        ObservabilitySinkType,
        issues,
        default=(ObservabilitySinkType.PGSQL,),
    )
    metrics = _optional_bool(
        raw, "observability.metrics", "metrics", issues, True
    )
    trace = _optional_bool(raw, "observability.trace", "trace", issues, True)
    capture_events = _optional_bool(
        raw,
        "observability.capture_events",
        "capture_events",
        issues,
        True,
    )
    if len(issues) > issue_count:
        return None
    assert sinks is not None
    assert metrics is not None
    assert trace is not None
    assert capture_events is not None
    return _construct(
        lambda: TaskObservabilityPolicy(
            sinks=sinks,
            metrics=metrics,
            trace=trace,
            capture_events=capture_events,
        ),
        "observability.invalid_value",
        "observability",
        issues,
    )


def _required_str(
    raw: RawSection,
    path: str,
    key: str,
    issues: list[TaskLoadIssue],
) -> str | None:
    value = raw.get(key)
    if value is None:
        issues.append(_missing_field(path))
        return None
    return _string_value(value, path, issues)


def _optional_str(
    raw: RawSection,
    path: str,
    key: str,
    issues: list[TaskLoadIssue],
) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    return _string_value(value, path, issues)


def _string_value(
    value: object, path: str, issues: list[TaskLoadIssue]
) -> str | None:
    if isinstance(value, str):
        return value
    issues.append(_invalid_type(path, "Use a string value."))
    return None


def _required_enum(
    raw: RawSection,
    path: str,
    key: str,
    enum_type: type[EnumValue],
    issues: list[TaskLoadIssue],
) -> EnumValue | None:
    value = raw.get(key)
    if value is None:
        issues.append(_missing_field(path))
        return None
    return _enum_value(value, path, enum_type, issues)


def _optional_enum(
    raw: RawSection,
    path: str,
    key: str,
    enum_type: type[EnumValue],
    issues: list[TaskLoadIssue],
) -> EnumValue | None:
    value = raw.get(key)
    if value is None:
        return None
    return _enum_value(value, path, enum_type, issues)


def _enum_value(
    value: object,
    path: str,
    enum_type: type[EnumValue],
    issues: list[TaskLoadIssue],
) -> EnumValue | None:
    if not isinstance(value, str):
        issues.append(_invalid_type(path, "Use a string enum value."))
        return None
    try:
        return enum_type(value)
    except ValueError:
        issues.append(
            _issue(
                code=_invalid_enum_code(path),
                path=path,
                message="Task definition enum value is not supported.",
                hint=(
                    "Use one of: "
                    + ", ".join(item.value for item in enum_type)
                    + "."
                ),
                category=TaskLoadIssueCategory.VALUE,
            )
        )
        return None


def _enum_tuple(
    raw: RawSection,
    path: str,
    key: str,
    enum_type: type[EnumValue],
    issues: list[TaskLoadIssue],
    *,
    default: tuple[EnumValue, ...],
) -> tuple[EnumValue, ...] | None:
    value = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, list):
        issues.append(_invalid_type(path, "Use an array of string values."))
        return None
    values: list[EnumValue] = []
    for index, item in enumerate(value):
        enum_value = _enum_value(item, f"{path}.{index}", enum_type, issues)
        if enum_value is not None:
            values.append(enum_value)
    if len(values) != len(value):
        return None
    return tuple(values)


def _optional_int(
    raw: RawSection,
    path: str,
    key: str,
    issues: list[TaskLoadIssue],
) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    issues.append(_invalid_type(path, "Use an integer value."))
    return None


def _optional_bool(
    raw: RawSection,
    path: str,
    key: str,
    issues: list[TaskLoadIssue],
    default: bool,
) -> bool | None:
    value = raw.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    issues.append(_invalid_type(path, "Use a boolean value."))
    return None


def _string_tuple(
    raw: RawSection,
    path: str,
    key: str,
    issues: list[TaskLoadIssue],
) -> tuple[str, ...]:
    value = raw.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        issues.append(_invalid_type(path, "Use an array of string values."))
        return ()
    values: list[str] = []
    for index, item in enumerate(value):
        string_value = _string_value(item, f"{path}.{index}", issues)
        if string_value is not None:
            values.append(string_value)
    return tuple(values)


def _optional_mapping(
    raw: RawSection,
    path: str,
    key: str,
    issues: list[TaskLoadIssue],
) -> FrozenMetadata | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, Mapping):
        return cast(FrozenMetadata, value)
    issues.append(_invalid_type(path, "Use a TOML table value."))
    return None


def _construct(
    build: Callable[[], DefinitionValue],
    code: str,
    path: str,
    issues: list[TaskLoadIssue],
) -> DefinitionValue | None:
    try:
        return build()
    except AssertionError:
        issues.append(
            _issue(
                code=code,
                path=path,
                message="Task definition values failed validation.",
                hint=(
                    "Check required fields, positive integer bounds, and "
                    "compatible section options."
                ),
                category=TaskLoadIssueCategory.VALUE,
            )
        )
        return None


def _skill_issue(diagnostic: SkillDiagnosticInfo) -> TaskLoadIssue:
    assert isinstance(diagnostic, SkillDiagnosticInfo)
    return _issue(
        code=diagnostic.code.value,
        path=_skill_diagnostic_path(diagnostic.path),
        message=diagnostic.message,
        hint=diagnostic.hint,
        category=TaskLoadIssueCategory.VALUE,
    )


def _skill_diagnostic_path(path: str) -> str:
    assert isinstance(path, str) and path.strip()
    suffix = path[len("settings.") :] if path.startswith("settings.") else path
    return f"skills.{suffix}"


def _assertion_hint(error: BaseException) -> str:
    message = str(error).strip()
    return message or "Use supported task skills settings."


def _missing_section(section: str) -> TaskLoadIssue:
    return _issue(
        code="task.missing_section",
        path=section,
        message="Task definition is missing a required section.",
        hint=f"Add a [{section}] table to the task definition.",
        category=TaskLoadIssueCategory.STRUCTURE,
    )


def _missing_field(path: str) -> TaskLoadIssue:
    return _issue(
        code="task.missing_field",
        path=path,
        message="Task definition is missing a required field.",
        hint="Add the required field to the task definition.",
        category=TaskLoadIssueCategory.STRUCTURE,
    )


def _invalid_type(path: str, hint: str) -> TaskLoadIssue:
    return _issue(
        code="task.invalid_type",
        path=path,
        message="Task definition field has the wrong type.",
        hint=hint,
        category=TaskLoadIssueCategory.VALUE,
    )


def _schema_resolution_issue(
    error: TaskSchemaResolutionError,
) -> TaskLoadIssue:
    return _issue(
        code=_schema_resolution_code(error.path),
        path=error.path,
        message="Task schema reference could not be resolved.",
        hint=(
            "Use a local JSON object schema file relative to the task "
            "definition."
        ),
        category=TaskLoadIssueCategory.VALUE,
    )


def _schema_resolution_code(path: str) -> str:
    match path.split(".", maxsplit=1)[0]:
        case "input":
            return "input.invalid_schema"
        case _:
            return "output.invalid_schema"


def _invalid_enum_code(path: str) -> str:
    match path.split(".", maxsplit=1)[0]:
        case "input":
            return "input.invalid_type"
        case "output":
            return "output.invalid_type"
        case "execution":
            return "execution.unknown_target"
        case "privacy":
            return "privacy.unknown_action"
        case "observability":
            return "observability.unsupported_sink"
        case _:
            return "task.invalid_enum"


def _source_path(source_path: str | Path | None) -> str:
    if source_path is None:
        return "toml"
    return str(source_path)


def _issue(
    *,
    code: str,
    path: str,
    message: str,
    hint: str,
    category: TaskLoadIssueCategory,
) -> TaskLoadIssue:
    return TaskLoadIssue(
        code=code,
        path=path,
        message=message,
        hint=hint,
        category=category,
    )
