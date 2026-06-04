from .artifact import (
    TaskArtifactPurpose,
    TaskArtifactRecord,
    TaskArtifactRef,
    TaskArtifactState,
    task_output_artifact_from_value,
)
from .converters import (
    FileConverter,
    TaskFileConversionDependencyError,
    TaskFileConversionError,
    validate_conversion_request,
)
from .definition import (
    PrivacyAction,
    TaskArtifactPolicy,
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskInputType,
    TaskLimitsPolicy,
    TaskOutputContract,
    TaskOutputType,
    TaskPrivacyPolicy,
    TaskRetryPolicy,
    TaskRunPolicy,
    TaskTargetType,
)
from .feature_gate import TaskFeature, feature_diagnostic
from .input import (
    TaskFileConversionRequest,
    TaskFileDescriptor,
    TaskFileSourceKind,
    TaskProviderReference,
    TaskProviderReferenceKind,
    TaskRemoteUrlPolicy,
)
from .privacy import (
    EncryptionProvider,
    HmacProvider,
    privacy_policy_fields,
    privacy_policy_hash_fields,
    privacy_policy_raw_fields,
    privacy_policy_store_fields,
)

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from importlib import import_module
from ipaddress import ip_address
from math import isfinite
from pathlib import Path, PurePosixPath, PureWindowsPath
from re import fullmatch
from types import ModuleType
from typing import Protocol, cast
from urllib.parse import urlsplit

_REQUIRED_SECTIONS = ("task", "input", "output", "execution")
_STRUCTURED_INPUT_TYPES = frozenset(
    {
        TaskInputType.OBJECT,
        TaskInputType.ARRAY,
    }
)
_STRUCTURED_OUTPUT_TYPES = frozenset(
    {
        TaskOutputType.JSON,
        TaskOutputType.OBJECT,
        TaskOutputType.ARRAY,
    }
)
_SUPPORTED_TARGET_TYPES = frozenset(
    {
        TaskTargetType.AGENT,
        TaskTargetType.FLOW,
    }
)
_PATH_TARGET_TYPES = frozenset(
    {
        TaskTargetType.AGENT,
        TaskTargetType.FLOW,
        TaskTargetType.TASK,
    }
)

TASK_FILE_DELIVERY_DIAGNOSTIC_CODES = frozenset(
    {
        "task.file_delivery.limit_exceeded",
        "task.file_delivery.missing_artifact_store",
        "task.file_delivery.missing_conversion",
        "task.file_delivery.provider_mismatch",
        "task.file_delivery.rejected",
        "task.file_delivery.unknown_size",
        "task.file_delivery.unsupported",
        "task.file_delivery.unsupported_mime",
    }
)
TASK_SECURITY_DIAGNOSTIC_CODES = frozenset(
    {
        "artifact.retention_violation",
        "input.unsafe_path",
        "privacy.raw_storage_policy_violation",
        "remote_url.ssrf_rejected",
    }
)
TASK_QUEUE_DIAGNOSTIC_CODES = frozenset(
    {
        "queue.file_payload_unavailable",
        "queue.input_payload_unavailable",
    }
)
TASK_VALIDATION_ISSUE_CODES = (
    frozenset(
        {
            "artifact.bytes_unsupported",
            "artifact.retention_required",
            "dependency.jsonschema_missing",
            "dependency.task_documents_missing",
            "execution.path_escape",
            "execution.unknown_target",
            "execution.unsupported_flow",
            "feature.flow_backed_tasks_disabled",
            "feature.remote_url_file_inputs_disabled",
            "input.invalid_file",
            "input.invalid_schema",
            "input.invalid_type",
            "limits.invalid_value",
            "observability.unsupported_sink",
            "output.invalid_schema",
            "output.invalid_type",
            "privacy.encryption_key_missing",
            "privacy.hmac_key_missing",
            "privacy.raw_retention_required",
            "privacy.unknown_action",
            "task.missing_section",
        }
    )
    | TASK_FILE_DELIVERY_DIAGNOSTIC_CODES
    | TASK_SECURITY_DIAGNOSTIC_CODES
)
TASK_VALIDATION_ISSUE_CODES = (
    TASK_VALIDATION_ISSUE_CODES | TASK_QUEUE_DIAGNOSTIC_CODES
)


class TaskValidationCategory(StrEnum):
    DEPENDENCY = "dependency"
    PRIVACY = "privacy"
    STRUCTURE = "structure"
    UNSUPPORTED = "unsupported"
    VALUE = "value"


class TaskValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class _JsonSchemaValidator(Protocol):
    def validate(self, instance: object) -> None: ...


class _JsonSchemaValidatorClass(Protocol):
    def __call__(
        self,
        schema: Mapping[str, object],
    ) -> _JsonSchemaValidator: ...

    def check_schema(self, schema: Mapping[str, object]) -> None: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class _JsonSchemaAdapter:
    validator_class: _JsonSchemaValidatorClass
    schema_error: type[Exception]
    validation_error: type[Exception]


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskValidationIssue:
    code: str
    path: str
    message: str
    hint: str
    category: TaskValidationCategory
    severity: TaskValidationSeverity = TaskValidationSeverity.ERROR

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "path": self.path,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "hint": self.hint,
        }


class TaskValidationError(ValueError):
    issues: tuple[TaskValidationIssue, ...]

    def __init__(self, issues: tuple[TaskValidationIssue, ...]) -> None:
        assert issues, "issues must not be empty"
        self.issues = issues
        summary = ", ".join(
            f"{issue.code} at {issue.path}" for issue in issues
        )
        super().__init__(f"task validation failed: {summary}")


def validate_task_definition(
    definition: TaskDefinition,
    *,
    hmac_provider: HmacProvider | None = None,
    encryption_provider: EncryptionProvider | None = None,
    require_configured_keys: bool = False,
    raw_storage_allowed: bool | None = None,
    execution_roots: Iterable[str | Path] = (),
    file_converters: Mapping[str, FileConverter] | None = None,
) -> tuple[TaskValidationIssue, ...]:
    assert isinstance(definition, TaskDefinition)
    issues: list[TaskValidationIssue] = []
    issues.extend(
        _validate_input_contract(
            definition.input,
            file_converters=file_converters,
        )
    )
    issues.extend(_validate_output_contract(definition.output))
    issues.extend(
        _validate_execution_target(
            definition.execution,
            execution_roots=tuple(execution_roots),
        )
    )
    issues.extend(
        _validate_privacy_policy(
            definition.privacy,
            definition.artifact,
            hmac_provider=hmac_provider,
            encryption_provider=encryption_provider,
            require_configured_keys=require_configured_keys,
            raw_storage_allowed=raw_storage_allowed,
        )
    )
    issues.extend(_validate_limits_policy(definition.run))
    issues.extend(_validate_limits_policy(definition.retry))
    issues.extend(_validate_limits_policy(definition.artifact))
    issues.extend(_validate_limits_policy(definition.limits))
    return tuple(issues)


def validate_task_sections(
    sections: Mapping[str, object],
) -> tuple[TaskValidationIssue, ...]:
    assert isinstance(sections, Mapping)
    issues: list[TaskValidationIssue] = []
    for section in _REQUIRED_SECTIONS:
        if section not in sections:
            issues.append(
                _issue(
                    code="task.missing_section",
                    path=section,
                    message="Task definition is missing a required section.",
                    hint=f"Add a [{section}] table to the task definition.",
                    category=TaskValidationCategory.STRUCTURE,
                )
            )
    return tuple(issues)


def validate_task_input(
    definition: TaskDefinition,
    value: object,
    *,
    remote_url_policy: TaskRemoteUrlPolicy | None = None,
    file_converters: Mapping[str, FileConverter] | None = None,
) -> tuple[TaskValidationIssue, ...]:
    assert isinstance(definition, TaskDefinition)
    issues = list(
        _validate_input_contract(
            definition.input,
            file_converters=file_converters,
        )
    )
    input_type = definition.input.type
    if not isinstance(input_type, TaskInputType):
        return tuple(issues)
    if value is None:
        if definition.input.required:
            issues.append(
                _invalid_input_type(
                    "input",
                    "Required task input is missing.",
                    "Pass a value matching the declared input contract.",
                )
            )
        return tuple(issues)

    match input_type:
        case TaskInputType.STRING:
            if not isinstance(value, str):
                issues.append(
                    _invalid_input_type(
                        "input",
                        "Task input must match the declared scalar type.",
                        "Pass a string value.",
                    )
                )
        case TaskInputType.INTEGER:
            if not _is_integer(value):
                issues.append(
                    _invalid_input_type(
                        "input",
                        "Task input must match the declared scalar type.",
                        "Pass an integer value.",
                    )
                )
        case TaskInputType.NUMBER:
            if not _is_number(value):
                issues.append(
                    _invalid_input_type(
                        "input",
                        "Task input must match the declared scalar type.",
                        "Pass a finite numeric value.",
                    )
                )
        case TaskInputType.BOOLEAN:
            if not isinstance(value, bool):
                issues.append(
                    _invalid_input_type(
                        "input",
                        "Task input must match the declared scalar type.",
                        "Pass a boolean value.",
                    )
                )
        case TaskInputType.OBJECT:
            if not isinstance(value, Mapping):
                issues.append(
                    _invalid_input_type(
                        "input",
                        "Task input must match the declared structured type.",
                        "Pass an object value.",
                    )
                )
            elif not _is_json_value(_structured_input_schema_value(value)):
                issues.append(
                    _invalid_input_type(
                        "input",
                        "Task input must be JSON compatible.",
                        "Pass an object with JSON-compatible values.",
                    )
                )
            else:
                issues.extend(
                    _validate_value_schema(
                        schema=definition.input.schema,
                        value=_structured_input_schema_value(value),
                        code="input.invalid_type",
                        path="input",
                        message="Task input does not match its schema.",
                        hint="Pass a value matching the declared schema.",
                    )
                )
                issues.extend(
                    _validate_structured_file_descriptors(
                        definition,
                        value,
                        path="input",
                        remote_url_policy=remote_url_policy,
                        file_converters=file_converters,
                    )
                )
        case TaskInputType.ARRAY:
            if not isinstance(value, list | tuple):
                issues.append(
                    _invalid_input_type(
                        "input",
                        "Task input must match the declared structured type.",
                        "Pass an array value.",
                    )
                )
            elif not _is_json_value(_structured_input_schema_value(value)):
                issues.append(
                    _invalid_input_type(
                        "input",
                        "Task input must be JSON compatible.",
                        "Pass an array with JSON-compatible values.",
                    )
                )
            else:
                issues.extend(
                    _validate_value_schema(
                        schema=definition.input.schema,
                        value=_structured_input_schema_value(value),
                        code="input.invalid_type",
                        path="input",
                        message="Task input does not match its schema.",
                        hint="Pass a value matching the declared schema.",
                    )
                )
                issues.extend(
                    _validate_structured_file_descriptors(
                        definition,
                        value,
                        path="input",
                        remote_url_policy=remote_url_policy,
                        file_converters=file_converters,
                    )
                )
        case TaskInputType.FILE:
            issues.extend(
                _validate_file_descriptor_value(
                    definition,
                    value,
                    path="input",
                    remote_url_policy=remote_url_policy,
                    file_converters=file_converters,
                )
            )
        case TaskInputType.FILE_ARRAY:
            if not isinstance(value, list | tuple):
                issues.append(
                    _invalid_input_type(
                        "input",
                        "Task input must match the declared file array type.",
                        "Pass an array of file descriptors.",
                    )
                )
            elif definition.input.required and not value:
                issues.append(
                    _invalid_file_issue(
                        "input",
                        "Task input requires at least one file descriptor.",
                        "Pass a non-empty array of file descriptors.",
                    )
                )
            else:
                issues.extend(
                    _validate_file_descriptor_count(definition, value)
                )
                for index, item in enumerate(value):
                    issues.extend(
                        _validate_file_descriptor_value(
                            definition,
                            item,
                            path=f"input[{index}]",
                            remote_url_policy=remote_url_policy,
                            file_converters=file_converters,
                        )
                    )
    return tuple(issues)


def validate_task_output(
    definition: TaskDefinition,
    value: object,
) -> tuple[TaskValidationIssue, ...]:
    assert isinstance(definition, TaskDefinition)
    issues = list(_validate_output_contract(definition.output))
    output_type = definition.output.type
    if not isinstance(output_type, TaskOutputType):
        return tuple(issues)

    match output_type:
        case TaskOutputType.TEXT:
            if not isinstance(value, str):
                issues.append(
                    _invalid_output_type(
                        "output",
                        "Task output must match the declared scalar type.",
                        "Return a text value.",
                    )
                )
        case TaskOutputType.OBJECT:
            if not isinstance(value, Mapping):
                issues.append(
                    _invalid_output_type(
                        "output",
                        "Task output must match the declared structured type.",
                        "Return an object value.",
                    )
                )
            elif not _is_json_value(value):
                issues.append(
                    _invalid_output_type(
                        "output",
                        "Task output must be JSON compatible.",
                        "Return an object with JSON-compatible values.",
                    )
                )
            else:
                issues.extend(
                    _validate_value_schema(
                        schema=definition.output.schema,
                        value=value,
                        code="output.invalid_type",
                        path="output",
                        message="Task output does not match its schema.",
                        hint="Return a value matching the declared schema.",
                    )
                )
        case TaskOutputType.ARRAY:
            if not isinstance(value, list | tuple):
                issues.append(
                    _invalid_output_type(
                        "output",
                        "Task output must match the declared structured type.",
                        "Return an array value.",
                    )
                )
            elif not _is_json_value(value):
                issues.append(
                    _invalid_output_type(
                        "output",
                        "Task output must be JSON compatible.",
                        "Return an array with JSON-compatible values.",
                    )
                )
            else:
                issues.extend(
                    _validate_value_schema(
                        schema=definition.output.schema,
                        value=value,
                        code="output.invalid_type",
                        path="output",
                        message="Task output does not match its schema.",
                        hint="Return a value matching the declared schema.",
                    )
                )
        case TaskOutputType.JSON:
            if not _is_json_value(value):
                issues.append(
                    _invalid_output_type(
                        "output",
                        "Task output must be JSON compatible.",
                        "Return a JSON-compatible value.",
                    )
                )
            else:
                issues.extend(
                    _validate_value_schema(
                        schema=definition.output.schema,
                        value=value,
                        code="output.invalid_type",
                        path="output",
                        message="Task output does not match its schema.",
                        hint="Return a value matching the declared schema.",
                    )
                )
        case TaskOutputType.FILE:
            issues.extend(
                _validate_output_artifact_value(
                    definition,
                    value,
                    path="output",
                    require_ready=True,
                )
            )
        case TaskOutputType.FILE_ARRAY | TaskOutputType.ARTIFACT_ARRAY:
            if not isinstance(value, list | tuple):
                issues.append(
                    _invalid_output_type(
                        "output",
                        "Task output must match the declared artifact type.",
                        "Return an array of task artifact references.",
                    )
                )
            else:
                issues.extend(
                    _validate_output_artifact_count(definition, value)
                )
                for index, item in enumerate(value):
                    issues.extend(
                        _validate_output_artifact_value(
                            definition,
                            item,
                            path=f"output[{index}]",
                            require_ready=(
                                output_type == TaskOutputType.FILE_ARRAY
                            ),
                        )
                    )
    return tuple(issues)


def _validate_output_artifact_count(
    definition: TaskDefinition,
    values: list[object] | tuple[object, ...],
) -> tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    for limit in (
        definition.artifact.max_count,
        definition.limits.artifact_count,
    ):
        if limit is not None and len(values) > limit:
            issues.append(
                _invalid_output_type(
                    "output",
                    "Task output exceeds the artifact count limit.",
                    "Return fewer output artifacts.",
                )
            )
            break
    return tuple(issues)


def _validate_output_artifact_value(
    definition: TaskDefinition,
    value: object,
    *,
    path: str,
    require_ready: bool,
) -> tuple[TaskValidationIssue, ...]:
    artifact = task_output_artifact_from_value(value)
    if artifact is None:
        return (
            _invalid_output_type(
                path,
                "Task output must be a task artifact reference.",
                "Return a TaskArtifactRef or output TaskArtifactRecord.",
            ),
        )
    issues: list[TaskValidationIssue] = []
    if require_ready and artifact.state != TaskArtifactState.READY:
        issues.append(
            _invalid_output_type(
                f"{path}.state",
                "Task file output must reference a ready artifact.",
                "Return ready file artifacts for file output contracts.",
            )
        )
    if (
        isinstance(value, TaskArtifactRecord)
        and value.purpose != TaskArtifactPurpose.OUTPUT
    ):
        issues.append(
            _invalid_output_type(
                f"{path}.purpose",
                "Task output artifact purpose is invalid.",
                "Return output artifact records.",
            )
        )
    ref = artifact.ref
    issues.extend(_validate_output_artifact_size(definition, ref, path=path))
    return tuple(issues)


def _validate_output_artifact_size(
    definition: TaskDefinition,
    ref: TaskArtifactRef,
    *,
    path: str,
) -> tuple[TaskValidationIssue, ...]:
    if ref.size_bytes is None:
        return ()
    for limit in (
        definition.artifact.max_bytes,
        definition.limits.artifact_bytes,
    ):
        if limit is not None and ref.size_bytes > limit:
            return (
                _invalid_output_type(
                    f"{path}.size_bytes",
                    "Task output artifact exceeds the byte limit.",
                    "Return smaller output artifacts.",
                ),
            )
    return ()


def raise_task_validation_error(
    issues: tuple[TaskValidationIssue, ...],
) -> None:
    if issues:
        raise TaskValidationError(issues)


def _validate_input_contract(
    contract: TaskInputContract,
    *,
    file_converters: Mapping[str, FileConverter] | None,
) -> tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    if not isinstance(contract.type, TaskInputType):
        issues.append(
            _invalid_input_type(
                "input.type",
                "Task input type is not supported.",
                "Use a supported input type.",
            )
        )
        return tuple(issues)

    issues.extend(
        _validate_schema_contract(
            schema=contract.schema,
            schema_ref=contract.schema_ref,
            structured=contract.type in _STRUCTURED_INPUT_TYPES,
            code="input.invalid_schema",
            path="input",
            expected_schema_type=_schema_type_for_input(contract.type),
        )
    )
    if contract.type in {TaskInputType.FILE, TaskInputType.FILE_ARRAY}:
        issues.extend(
            _validate_file_contract(
                contract,
                file_converters=file_converters,
            )
        )
    return tuple(issues)


def _validate_file_contract(
    contract: TaskInputContract,
    *,
    file_converters: Mapping[str, FileConverter] | None,
) -> tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    for index, mime_type in enumerate(contract.mime_types):
        if not _is_valid_mime_type(mime_type):
            issues.append(
                _invalid_file_issue(
                    f"input.mime_types[{index}]",
                    "Task file MIME type filter is invalid.",
                    "Use MIME type values such as text/plain.",
                )
            )
    for index, conversion in enumerate(contract.file_conversions):
        if not _is_valid_file_token(conversion):
            issues.append(
                _invalid_file_issue(
                    f"input.file_conversions[{index}]",
                    "Task file conversion name is invalid.",
                    "Use a stable conversion name.",
                )
            )
            continue
        issues.extend(
            _validate_file_conversion_capability(
                conversion,
                None,
                source_media_type=None,
                source_size_bytes=None,
                path=f"input.file_conversions[{index}]",
                file_converters=file_converters,
            )
        )
    return tuple(issues)


def _validate_file_descriptor_count(
    definition: TaskDefinition,
    values: list[object] | tuple[object, ...],
) -> tuple[TaskValidationIssue, ...]:
    limit = definition.limits.file_count
    if limit is None or len(values) <= limit:
        return ()
    return (
        _invalid_file_issue(
            "input",
            "Task input exceeds the file count limit.",
            "Pass fewer file descriptors.",
        ),
    )


def _validate_file_descriptor_value(
    definition: TaskDefinition,
    value: object,
    *,
    path: str,
    remote_url_policy: TaskRemoteUrlPolicy | None,
    file_converters: Mapping[str, FileConverter] | None,
) -> tuple[TaskValidationIssue, ...]:
    descriptor = _file_descriptor_mapping(value)
    if descriptor is None:
        return (
            _invalid_file_issue(
                path,
                "Task input must be a file descriptor.",
                "Pass a file descriptor with source_kind and reference.",
            ),
        )

    issues: list[TaskValidationIssue] = []
    try:
        source_kind = descriptor.get("source_kind")
        reference = descriptor.get("reference")
        role = descriptor.get("role")
        mime_type = descriptor.get("mime_type")
        size_bytes = descriptor.get("size_bytes")
        sha256 = descriptor.get("sha256")
        conversions = descriptor.get("conversions", ())
        metadata = descriptor.get("metadata")
        provider_reference = descriptor.get("provider_reference")
    except Exception:
        return (
            _invalid_file_issue(
                path,
                "Task input must be a file descriptor.",
                "Pass a file descriptor with source_kind and reference.",
            ),
        )

    if not _is_valid_source_kind(source_kind):
        issues.append(
            _invalid_file_issue(
                f"{path}.source_kind",
                "Task file source kind is invalid.",
                "Use a supported file source kind.",
            )
        )
    source_kind_value = _source_kind_value(source_kind)
    if not isinstance(reference, str) or not reference.strip():
        issues.append(
            _invalid_file_issue(
                f"{path}.reference",
                "Task file reference is invalid.",
                "Pass a non-empty file reference.",
            )
        )
    elif source_kind_value == TaskFileSourceKind.REMOTE_URL:
        issues.extend(
            _validate_remote_url_descriptor(
                definition,
                reference,
                path=path,
                policy=remote_url_policy,
            )
        )
    elif source_kind_value == TaskFileSourceKind.PROVIDER_REFERENCE:
        issues.extend(
            _validate_provider_reference_descriptor(
                reference,
                provider_reference,
                conversions,
                mime_type,
                path=path,
            )
        )
    elif provider_reference is not None:
        issues.append(
            _invalid_file_issue(
                f"{path}.provider_reference",
                "Task file provider reference is not allowed.",
                "Use provider_reference as the file source kind.",
            )
        )
    if role is not None and (
        not isinstance(role, str) or not _is_valid_file_token(role)
    ):
        issues.append(
            _invalid_file_issue(
                f"{path}.role",
                "Task file role is invalid.",
                "Use a stable role token.",
            )
        )
    if mime_type is not None and (
        not isinstance(mime_type, str) or not _is_valid_mime_type(mime_type)
    ):
        issues.append(
            _invalid_file_issue(
                f"{path}.mime_type",
                "Task file MIME type is invalid.",
                "Use MIME type values such as text/plain.",
            )
        )
    if (
        definition.input.mime_types
        and mime_type not in definition.input.mime_types
    ):
        issues.append(
            _invalid_file_issue(
                f"{path}.mime_type",
                "Task file MIME type is not allowed by the input contract.",
                "Pass a file whose MIME type is allowed by the contract.",
            )
        )
    if not _is_non_negative_int(size_bytes, optional=True):
        issues.append(
            _invalid_file_issue(
                f"{path}.size_bytes",
                "Task file size is invalid.",
                "Use a non-negative byte count.",
            )
        )
    elif (
        isinstance(size_bytes, int)
        and definition.limits.file_bytes is not None
        and size_bytes > definition.limits.file_bytes
    ):
        issues.append(
            _invalid_file_issue(
                f"{path}.size_bytes",
                "Task file exceeds the file byte limit.",
                "Pass a smaller file descriptor.",
            )
        )
    if sha256 is not None and (
        not isinstance(sha256, str) or not _is_valid_sha256(sha256)
    ):
        issues.append(
            _invalid_file_issue(
                f"{path}.sha256",
                "Task file digest is invalid.",
                "Use a lowercase SHA-256 hex digest.",
            )
        )
    issues.extend(
        _validate_file_conversions(
            definition,
            conversions,
            path=f"{path}.conversions",
            file_converters=file_converters,
            source_media_type=(
                mime_type if isinstance(mime_type, str) else None
            ),
            source_size_bytes=(
                size_bytes
                if isinstance(size_bytes, int)
                and not isinstance(size_bytes, bool)
                else None
            ),
        )
    )
    if metadata is not None and (
        not isinstance(metadata, Mapping) or not _is_json_value(metadata)
    ):
        issues.append(
            _invalid_file_issue(
                f"{path}.metadata",
                "Task file metadata is invalid.",
                "Use JSON-compatible metadata with string keys.",
            )
        )
    return tuple(issues)


def _file_descriptor_mapping(
    value: object,
) -> Mapping[str, object] | None:
    if isinstance(value, TaskFileDescriptor):
        return {
            "source_kind": value.source_kind,
            "reference": value.reference,
            "role": value.role,
            "mime_type": value.mime_type,
            "size_bytes": value.size_bytes,
            "sha256": value.sha256,
            "conversions": value.conversions,
            "metadata": value.metadata,
            "provider_reference": value.provider_reference,
        }
    if isinstance(value, Mapping):
        return value
    return None


def _validate_structured_file_descriptors(
    definition: TaskDefinition,
    value: object,
    *,
    path: str,
    remote_url_policy: TaskRemoteUrlPolicy | None,
    file_converters: Mapping[str, FileConverter] | None,
) -> tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    _collect_structured_file_descriptor_issues(
        definition,
        value,
        path=path,
        issues=issues,
        remote_url_policy=remote_url_policy,
        file_converters=file_converters,
    )
    return tuple(issues)


def _collect_structured_file_descriptor_issues(
    definition: TaskDefinition,
    value: object,
    *,
    path: str,
    issues: list[TaskValidationIssue],
    remote_url_policy: TaskRemoteUrlPolicy | None,
    file_converters: Mapping[str, FileConverter] | None,
) -> None:
    try:
        descriptor = _file_descriptor_mapping(value)
    except Exception:
        issues.append(
            _invalid_file_issue(
                path,
                "Task input must be a valid file descriptor.",
                "Pass a file descriptor with safe metadata values.",
            )
        )
        return
    if descriptor is not None and "source_kind" in descriptor:
        issues.extend(
            _validate_file_descriptor_value(
                definition,
                value,
                path=path,
                remote_url_policy=remote_url_policy,
                file_converters=file_converters,
            )
        )
        return
    if isinstance(value, Mapping):
        try:
            items = tuple(value.items())
        except Exception:
            issues.append(
                _invalid_file_issue(
                    path,
                    "Task input must be a valid file descriptor.",
                    "Pass a file descriptor with safe metadata values.",
                )
            )
            return
        for key, item in items:
            if not isinstance(key, str):
                continue
            _collect_structured_file_descriptor_issues(
                definition,
                item,
                path=f"{path}.{key}",
                issues=issues,
                remote_url_policy=remote_url_policy,
                file_converters=file_converters,
            )
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _collect_structured_file_descriptor_issues(
                definition,
                item,
                path=f"{path}[{index}]",
                issues=issues,
                remote_url_policy=remote_url_policy,
                file_converters=file_converters,
            )


def _validate_provider_reference_descriptor(
    reference: object,
    value: object,
    conversions: object,
    descriptor_mime_type: object,
    *,
    path: str,
) -> tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    if conversions:
        issues.append(
            _invalid_file_issue(
                f"{path}.conversions",
                "Task file conversion conflicts with provider reference.",
                "Use either a provider reference or a conversion request.",
            )
        )
    if isinstance(value, TaskProviderReference):
        if value.reference != reference:
            issues.append(
                _invalid_file_issue(
                    f"{path}.provider_reference.reference",
                    "Task file provider reference does not match.",
                    "Use the same reference on the descriptor and provider.",
                )
            )
        if _provider_reference_mime_mismatch(
            descriptor_mime_type,
            value.mime_type,
        ):
            issues.append(
                _invalid_file_issue(
                    f"{path}.provider_reference.mime_type",
                    "Task file provider reference MIME type does not match.",
                    "Use one MIME type for the descriptor and provider.",
                )
            )
        return tuple(issues)
    if not isinstance(value, Mapping):
        issues.append(
            _invalid_file_issue(
                f"{path}.provider_reference",
                "Task file provider reference is invalid.",
                "Pass provider, kind, durability, and identity metadata.",
            )
        )
        return tuple(issues)
    kind = value.get("kind")
    provider = value.get("provider")
    provider_reference = value.get("reference")
    durable = value.get("durable", True)
    expires_at = value.get("expires_at")
    owner_scope = value.get("owner_scope")
    mime_type = value.get("mime_type")
    size_bucket = value.get("size_bucket")
    identity_hmac = value.get("identity_hmac")
    metadata = value.get("metadata")
    if not _is_valid_provider_reference_kind(kind):
        issues.append(
            _invalid_file_issue(
                f"{path}.provider_reference.kind",
                "Task file provider reference kind is invalid.",
                "Use a supported provider reference kind.",
            )
        )
    if not isinstance(provider, str) or not _is_valid_file_token(provider):
        issues.append(
            _invalid_file_issue(
                f"{path}.provider_reference.provider",
                "Task file provider reference provider is invalid.",
                "Use a stable provider token.",
            )
        )
    if provider_reference != reference:
        issues.append(
            _invalid_file_issue(
                f"{path}.provider_reference.reference",
                "Task file provider reference does not match.",
                "Use the same reference on the descriptor and provider.",
            )
        )
    if not isinstance(durable, bool):
        issues.append(
            _invalid_file_issue(
                f"{path}.provider_reference.durable",
                "Task file provider reference durability is invalid.",
                "Use true for durable references and false otherwise.",
            )
        )
    if expires_at is not None and not _valid_provider_reference_expires_at(
        expires_at
    ):
        issues.append(
            _invalid_file_issue(
                f"{path}.provider_reference.expires_at",
                "Task file provider reference expiry is invalid.",
                "Use an ISO 8601 timestamp with timezone.",
            )
        )
    for field_name, field_value in (
        ("owner_scope", owner_scope),
        ("size_bucket", size_bucket),
        ("identity_hmac", identity_hmac),
    ):
        if field_value is not None and (
            not isinstance(field_value, str) or not field_value.strip()
        ):
            issues.append(
                _invalid_file_issue(
                    f"{path}.provider_reference.{field_name}",
                    "Task file provider reference metadata is invalid.",
                    "Use non-empty string metadata values.",
                )
            )
    if mime_type is not None and (
        not isinstance(mime_type, str) or not _is_valid_mime_type(mime_type)
    ):
        issues.append(
            _invalid_file_issue(
                f"{path}.provider_reference.mime_type",
                "Task file provider reference MIME type is invalid.",
                "Use MIME type values such as application/pdf.",
            )
        )
    elif _provider_reference_mime_mismatch(descriptor_mime_type, mime_type):
        issues.append(
            _invalid_file_issue(
                f"{path}.provider_reference.mime_type",
                "Task file provider reference MIME type does not match.",
                "Use one MIME type for the descriptor and provider.",
            )
        )
    if metadata is not None and (
        not isinstance(metadata, Mapping) or not _is_json_value(metadata)
    ):
        issues.append(
            _invalid_file_issue(
                f"{path}.provider_reference.metadata",
                "Task file provider reference metadata is invalid.",
                "Use JSON-compatible metadata with string keys.",
            )
        )
    return tuple(issues)


def _provider_reference_mime_mismatch(
    descriptor_mime_type: object,
    provider_mime_type: object,
) -> bool:
    return (
        isinstance(descriptor_mime_type, str)
        and isinstance(provider_mime_type, str)
        and descriptor_mime_type != provider_mime_type
    )


def _is_valid_provider_reference_kind(value: object) -> bool:
    if isinstance(value, TaskProviderReferenceKind):
        return True
    if not isinstance(value, str):
        return False
    try:
        TaskProviderReferenceKind(value)
    except ValueError:
        return False
    return True


def _valid_provider_reference_expires_at(value: object) -> bool:
    if isinstance(value, datetime):
        return value.tzinfo is not None
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _is_valid_source_kind(value: object) -> bool:
    if isinstance(value, TaskFileSourceKind):
        return True
    if not isinstance(value, str):
        return False
    try:
        TaskFileSourceKind(value)
    except ValueError:
        return False
    return True


def _source_kind_value(value: object) -> TaskFileSourceKind | None:
    if isinstance(value, TaskFileSourceKind):
        return value
    if not isinstance(value, str):
        return None
    try:
        return TaskFileSourceKind(value)
    except ValueError:
        return None


def _validate_remote_url_descriptor(
    definition: TaskDefinition,
    reference: str,
    *,
    path: str,
    policy: TaskRemoteUrlPolicy | None,
) -> tuple[TaskValidationIssue, ...]:
    if policy is None or not policy.enabled:
        return (
            _feature_issue(
                TaskFeature.REMOTE_URL_FILE_INPUTS,
                path=f"{path}.source_kind",
            ),
        )

    issues: list[TaskValidationIssue] = []
    try:
        parsed = urlsplit(reference)
        hostname = parsed.hostname
        _port = parsed.port
    except ValueError:
        return (
            _invalid_file_issue(
                f"{path}.reference",
                "Remote URL file reference is invalid.",
                "Use a valid absolute URL.",
            ),
        )

    scheme = parsed.scheme.lower()
    if scheme not in policy.allowed_schemes:
        issues.append(
            _invalid_file_issue(
                f"{path}.reference",
                "Remote URL file scheme is not allowed.",
                "Use an allowed remote URL scheme.",
            )
        )
    if not hostname:
        issues.append(
            _invalid_file_issue(
                f"{path}.reference",
                "Remote URL file host is invalid.",
                "Use an absolute URL with a host.",
            )
        )
    elif not policy.allow_private_networks and _is_private_network_host(
        hostname
    ):
        issues.append(
            _invalid_file_issue(
                f"{path}.reference",
                "Remote URL file host is not allowed.",
                "Use a public remote host or disable remote URL inputs.",
            )
        )
    if parsed.username is not None or parsed.password is not None:
        issues.append(
            _invalid_file_issue(
                f"{path}.reference",
                "Remote URL file credentials are not allowed.",
                "Pass credentials through a configured secret provider.",
            )
        )
    if policy.allow_redirects and policy.max_redirects <= 0:
        issues.append(
            _invalid_file_issue(
                f"{path}.redirects",
                "Remote URL redirects require a redirect limit.",
                "Set a positive remote URL redirect limit.",
            )
        )
    if (
        policy.max_bytes is None
        and definition.limits.file_bytes is None
        and definition.artifact.max_bytes is None
    ):
        issues.append(
            _invalid_file_issue(
                f"{path}.size_bytes",
                "Remote URL file inputs require a byte limit.",
                "Set a remote URL, task file, or artifact byte limit.",
            )
        )
    return tuple(issues)


def _is_private_network_host(hostname: str) -> bool:
    normalized = hostname.rstrip(".").lower()
    if (
        normalized == "localhost"
        or normalized.endswith(".localhost")
        or normalized.endswith(".local")
    ):
        return True
    try:
        address = ip_address(normalized)
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _validate_file_conversions(
    definition: TaskDefinition,
    value: object,
    *,
    path: str,
    file_converters: Mapping[str, FileConverter] | None,
    source_media_type: str | None,
    source_size_bytes: int | None,
) -> tuple[TaskValidationIssue, ...]:
    if not isinstance(value, list | tuple):
        return (
            _invalid_file_issue(
                path,
                "Task file conversions are invalid.",
                "Pass conversion requests as an array.",
            ),
        )
    issues: list[TaskValidationIssue] = []
    for index, conversion in enumerate(value):
        name = _file_conversion_name(conversion)
        if name is None or not _is_valid_file_token(name):
            issues.append(
                _invalid_file_issue(
                    f"{path}[{index}]",
                    "Task file conversion request is invalid.",
                    "Use a stable conversion name.",
                )
            )
            continue
        if (
            definition.input.file_conversions
            and name not in definition.input.file_conversions
        ):
            issues.append(
                _invalid_file_issue(
                    f"{path}[{index}]",
                    "Task file conversion is not allowed by the contract.",
                    "Use a conversion declared by the input contract.",
                )
            )
        options = _file_conversion_options(conversion)
        if options is not None and (
            not isinstance(options, Mapping) or not _is_json_value(options)
        ):
            issues.append(
                _invalid_file_issue(
                    f"{path}[{index}].options",
                    "Task file conversion options are invalid.",
                    "Use JSON-compatible conversion options.",
                )
            )
            continue
        issues.extend(
            _validate_file_conversion_capability(
                name,
                options,
                source_media_type=source_media_type,
                source_size_bytes=source_size_bytes,
                path=f"{path}[{index}]",
                file_converters=file_converters,
            )
        )
    return tuple(issues)


def _validate_file_conversion_capability(
    name: str,
    options: object,
    *,
    source_media_type: str | None,
    source_size_bytes: int | None,
    path: str,
    file_converters: Mapping[str, FileConverter] | None,
) -> tuple[TaskValidationIssue, ...]:
    if name in {"native", "none"}:
        return ()
    if file_converters is None:
        return ()
    converter = file_converters.get(name)
    if converter is None:
        return (
            _invalid_file_issue(
                path,
                "Task file conversion is not supported.",
                "Register a converter with this name or use a supported"
                " conversion.",
            ),
        )
    try:
        validate_conversion_request(
            converter,
            TaskFileConversionRequest(
                name=name,
                options=(options if isinstance(options, Mapping) else {}),
            ),
            source_media_type=source_media_type,
            source_size_bytes=source_size_bytes,
        )
    except TaskFileConversionDependencyError as error:
        return (_feature_issue(error.feature, path=path),)
    except TaskFileConversionError:
        return (
            _invalid_file_issue(
                path,
                "Task file conversion is not compatible with this input.",
                "Use a conversion that supports the MIME type, size, and"
                " options.",
            ),
        )
    return ()


def _file_conversion_name(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, TaskFileConversionRequest):
        return value.name
    if isinstance(value, Mapping):
        name = value.get("name")
        if isinstance(name, str):
            return name
    return None


def _file_conversion_options(value: object) -> object:
    if isinstance(value, TaskFileConversionRequest):
        return value.options
    if isinstance(value, Mapping):
        return value.get("options")
    return None


def _is_valid_mime_type(value: str) -> bool:
    return bool(
        fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}/"
            r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}",
            value,
        )
    )


def _is_valid_file_token(value: str) -> bool:
    return bool(fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", value))


def _is_non_negative_int(value: object, *, optional: bool) -> bool:
    if value is None and optional:
        return True
    return (
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
    )


def _is_valid_sha256(value: str) -> bool:
    return bool(fullmatch(r"[a-f0-9]{64}", value))


def _validate_output_contract(
    contract: TaskOutputContract,
) -> tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    if not isinstance(contract.type, TaskOutputType):
        issues.append(
            _invalid_output_type(
                "output.type",
                "Task output type is not supported.",
                "Use a supported output type.",
            )
        )
        return tuple(issues)

    issues.extend(
        _validate_schema_contract(
            schema=contract.schema,
            schema_ref=contract.schema_ref,
            structured=contract.type in _STRUCTURED_OUTPUT_TYPES,
            code="output.invalid_schema",
            path="output",
            expected_schema_type=_schema_type_for_output(contract.type),
        )
    )
    return tuple(issues)


def _validate_schema_contract(
    *,
    schema: object,
    schema_ref: object,
    structured: bool,
    code: str,
    path: str,
    expected_schema_type: str | None,
) -> tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    schema_path = f"{path}.schema"
    schema_ref_path = f"{path}.schema_ref"

    if not structured and (schema is not None or schema_ref is not None):
        issues.append(
            _invalid_schema_issue(
                code,
                schema_path,
                "Schemas are only valid for structured task contracts.",
                "Remove the schema or change the contract to a structured"
                " type.",
            )
        )
        return tuple(issues)

    if not structured:
        return ()

    if schema is None and schema_ref is None:
        issues.append(
            _invalid_schema_issue(
                code,
                schema_path,
                "Structured task contracts require a schema.",
                "Provide an inline schema or schema_ref.",
            )
        )
        return tuple(issues)

    if schema is not None and schema_ref is not None:
        issues.append(
            _invalid_schema_issue(
                code,
                schema_path,
                "Task contracts cannot declare both schema forms.",
                "Use either schema or schema_ref, not both.",
            )
        )

    if schema is not None:
        if not isinstance(schema, Mapping):
            issues.append(
                _invalid_schema_issue(
                    code,
                    schema_path,
                    "Task contract schema must be a JSON object.",
                    "Provide a JSON-compatible object schema.",
                )
            )
        elif not _is_json_value(schema):
            issues.append(
                _invalid_schema_issue(
                    code,
                    schema_path,
                    "Task contract schema must be JSON compatible.",
                    "Use only JSON-compatible schema values.",
                )
            )
        else:
            declared_type_issues = _validate_schema_declared_type(
                schema,
                expected_schema_type,
                code,
                schema_path,
            )
            issues.extend(declared_type_issues)
            if not declared_type_issues:
                issues.extend(
                    _validate_schema_syntax(
                        schema,
                        code=code,
                        path=schema_path,
                    )
                )

    if schema_ref is not None and (
        not isinstance(schema_ref, str) or not schema_ref.strip()
    ):
        issues.append(
            _invalid_schema_issue(
                code,
                schema_ref_path,
                "Task contract schema reference is invalid.",
                "Use a non-empty logical schema reference.",
            )
        )

    return tuple(issues)


def _validate_schema_syntax(
    schema: Mapping[object, object],
    *,
    code: str,
    path: str,
) -> tuple[TaskValidationIssue, ...]:
    adapter = _json_schema_adapter()
    if adapter is None:
        return (_json_schema_missing_issue(path),)
    schema_data = cast(Mapping[str, object], _json_compatible_data(schema))
    try:
        adapter.validator_class.check_schema(schema_data)
    except adapter.schema_error:
        return (
            _invalid_schema_issue(
                code,
                path,
                "Task contract schema is not valid JSON Schema.",
                "Use a valid JSON Schema Draft 2020-12 schema.",
            ),
        )
    return ()


def _validate_value_schema(
    *,
    schema: Mapping[str, object] | None,
    value: object,
    code: str,
    path: str,
    message: str,
    hint: str,
) -> tuple[TaskValidationIssue, ...]:
    if schema is None:
        return ()
    adapter = _json_schema_adapter()
    if adapter is None:
        return (_json_schema_missing_issue(f"{path}.schema"),)
    schema_data = cast(Mapping[str, object], _json_compatible_data(schema))
    value_data = _json_compatible_data(value)
    try:
        validator = adapter.validator_class(schema_data)
        validator.validate(value_data)
    except adapter.schema_error:
        return (
            _invalid_schema_issue(
                code.replace("invalid_type", "invalid_schema"),
                f"{path}.schema",
                "Task contract schema is not valid JSON Schema.",
                "Use a valid JSON Schema Draft 2020-12 schema.",
            ),
        )
    except adapter.validation_error:
        return (
            _issue(
                code=code,
                path=path,
                message=message,
                hint=hint,
                category=TaskValidationCategory.VALUE,
            ),
        )
    return ()


def _validate_schema_declared_type(
    schema: Mapping[object, object],
    expected_schema_type: str | None,
    code: str,
    path: str,
) -> tuple[TaskValidationIssue, ...]:
    if expected_schema_type is None or "type" not in schema:
        return ()
    declared_type = schema["type"]
    if declared_type == expected_schema_type:
        return ()
    if (
        isinstance(declared_type, list | tuple)
        and expected_schema_type in declared_type
    ):
        return ()
    return (
        _invalid_schema_issue(
            code,
            path,
            "Task contract schema type does not match the contract.",
            "Use a schema type that matches the task contract type.",
        ),
    )


def _validate_execution_target(
    target: TaskExecutionTarget,
    *,
    execution_roots: tuple[str | Path, ...],
) -> tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    if not isinstance(target.type, TaskTargetType):
        return (
            _issue(
                code="execution.unknown_target",
                path="execution.type",
                message="Task execution target is not supported.",
                hint="Use a supported execution target.",
                category=TaskValidationCategory.UNSUPPORTED,
            ),
        )
    if target.type in _PATH_TARGET_TYPES:
        issues.extend(
            _validate_execution_ref(
                target.ref,
                execution_roots=execution_roots,
            )
        )
    if target.type not in _SUPPORTED_TARGET_TYPES:
        issues.append(
            _issue(
                code="execution.unknown_target",
                path="execution.type",
                message="Task execution target is not supported.",
                hint="Use an agent execution target.",
                category=TaskValidationCategory.UNSUPPORTED,
            )
        )
    return tuple(issues)


def _validate_execution_ref(
    ref: object,
    *,
    execution_roots: tuple[str | Path, ...],
) -> tuple[TaskValidationIssue, ...]:
    if not isinstance(ref, str) or not ref.strip():
        return (_path_escape_issue(),)
    if _is_path_escape(ref):
        return (_path_escape_issue(),)
    roots = tuple(Path(root).resolve(strict=False) for root in execution_roots)
    if not roots:
        return ()
    for root in roots:
        try:
            candidate = (root / ref).resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if _is_relative_to(candidate, root):
            return ()
    return (_path_escape_issue(),)


def _validate_privacy_policy(
    policy: object,
    artifact: TaskArtifactPolicy,
    *,
    hmac_provider: HmacProvider | None,
    encryption_provider: EncryptionProvider | None,
    require_configured_keys: bool,
    raw_storage_allowed: bool | None,
) -> tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    invalid_action_paths = set()
    if not isinstance(policy, TaskPrivacyPolicy):
        return (
            _issue(
                code="privacy.unknown_action",
                path="privacy",
                message="Task privacy policy is invalid.",
                hint="Use supported privacy actions.",
                category=TaskValidationCategory.PRIVACY,
            ),
        )
    for field_name, action in privacy_policy_fields(policy).items():
        if not isinstance(action, PrivacyAction):
            invalid_action_paths.add(f"privacy.{field_name}")
            issues.append(
                _issue(
                    code="privacy.unknown_action",
                    path=f"privacy.{field_name}",
                    message="Task privacy action is not supported.",
                    hint="Use drop, hash, redact, store, or encrypt.",
                    category=TaskValidationCategory.PRIVACY,
                )
            )

    raw_retention_days = getattr(policy, "raw_retention_days")
    if (
        not isinstance(raw_retention_days, int)
        or isinstance(raw_retention_days, bool)
        or raw_retention_days < 0
    ):
        issues.append(
            _invalid_limit_issue(
                "privacy.raw_retention_days",
                "Use a non-negative integer value.",
            )
        )
        raw_retention_days = 0

    raw_fields = tuple(
        field_name
        for field_name in privacy_policy_raw_fields(policy)
        if f"privacy.{field_name}" not in invalid_action_paths
    )
    if raw_fields and raw_retention_days <= 0:
        issues.append(
            _issue(
                code="privacy.raw_retention_required",
                path="privacy.raw_retention_days",
                message="Raw privacy storage requires positive retention.",
                hint=(
                    "Set a positive raw retention period or use a non-raw "
                    "privacy action."
                ),
                category=TaskValidationCategory.PRIVACY,
            )
        )
    if raw_fields and raw_storage_allowed is False:
        issues.append(
            _issue(
                code="privacy.raw_retention_required",
                path="privacy",
                message="Raw privacy storage is not enabled.",
                hint=(
                    "Enable raw storage in runtime configuration or use a "
                    "non-raw privacy action."
                ),
                category=TaskValidationCategory.PRIVACY,
            )
        )
    for field_name in privacy_policy_store_fields(policy):
        path = f"privacy.{field_name}"
        issues.append(
            _issue(
                code="privacy.encryption_key_missing",
                path=path,
                message="Raw privacy storage must be encrypted.",
                hint="Use encrypt or a non-raw privacy action.",
                category=TaskValidationCategory.PRIVACY,
            )
        )
    if require_configured_keys:
        hash_fields = privacy_policy_hash_fields(policy)
        if hash_fields and hmac_provider is None:
            issues.append(
                _issue(
                    code="privacy.hmac_key_missing",
                    path=f"privacy.{hash_fields[0]}",
                    message="Privacy hashing requires a configured HMAC key.",
                    hint=(
                        "Configure a task HMAC key provider before validation."
                    ),
                    category=TaskValidationCategory.PRIVACY,
                )
            )
        encrypted_fields = tuple(
            field_name
            for field_name, action in privacy_policy_fields(policy).items()
            if action == PrivacyAction.ENCRYPT
        )
        if encrypted_fields and encryption_provider is None:
            issues.append(
                _issue(
                    code="privacy.encryption_key_missing",
                    path=f"privacy.{encrypted_fields[0]}",
                    message=(
                        "Encrypted privacy storage requires a configured "
                        "encryption key."
                    ),
                    hint=(
                        "Configure a task encryption provider before "
                        "validation."
                    ),
                    category=TaskValidationCategory.PRIVACY,
                )
            )
    issues.extend(
        _validate_artifact_raw_storage(
            artifact,
            encryption_provider=encryption_provider,
            require_configured_keys=require_configured_keys,
            raw_storage_allowed=raw_storage_allowed,
        )
    )
    return tuple(issues)


def _validate_artifact_raw_storage(
    artifact: TaskArtifactPolicy,
    *,
    encryption_provider: EncryptionProvider | None,
    require_configured_keys: bool,
    raw_storage_allowed: bool | None,
) -> tuple[TaskValidationIssue, ...]:
    if not artifact.store_bytes:
        return ()
    issues: list[TaskValidationIssue] = []
    if (
        not isinstance(artifact.retention_days, int)
        or isinstance(artifact.retention_days, bool)
        or artifact.retention_days <= 0
    ):
        issues.append(
            _issue(
                code="artifact.retention_required",
                path="artifact.retention_days",
                message="Artifact byte storage requires positive retention.",
                hint=(
                    "Set positive artifact retention or store references only."
                ),
                category=TaskValidationCategory.PRIVACY,
            )
        )
    if raw_storage_allowed is False:
        issues.append(
            _issue(
                code="artifact.bytes_unsupported",
                path="artifact.store_bytes",
                message="Artifact byte storage is not enabled.",
                hint=(
                    "Enable artifact byte storage in runtime configuration "
                    "or store references only."
                ),
                category=TaskValidationCategory.UNSUPPORTED,
            )
        )
    if not artifact.encrypt:
        issues.append(
            _issue(
                code="privacy.encryption_key_missing",
                path="artifact.encrypt",
                message="Artifact byte storage must be encrypted.",
                hint="Enable artifact encryption or store references only.",
                category=TaskValidationCategory.PRIVACY,
            )
        )
    elif require_configured_keys and encryption_provider is None:
        issues.append(
            _issue(
                code="privacy.encryption_key_missing",
                path="artifact.encrypt",
                message=(
                    "Artifact byte storage requires a configured "
                    "encryption key."
                ),
                hint="Configure a task encryption provider before validation.",
                category=TaskValidationCategory.PRIVACY,
            )
        )
    return tuple(issues)


def _json_schema_adapter() -> _JsonSchemaAdapter | None:
    try:
        module = import_module("jsonschema")
    except (ImportError, ValueError):
        return None
    return _json_schema_adapter_from_module(module)


def _json_schema_adapter_from_module(
    module: ModuleType,
) -> _JsonSchemaAdapter | None:
    validator_class = getattr(module, "Draft202012Validator", None)
    schema_error = _exception_class(module, "SchemaError")
    validation_error = _exception_class(module, "ValidationError")
    if (
        validator_class is None
        or schema_error is None
        or validation_error is None
    ):
        return None
    return _JsonSchemaAdapter(
        validator_class=cast(_JsonSchemaValidatorClass, validator_class),
        schema_error=schema_error,
        validation_error=validation_error,
    )


def _exception_class(
    module: ModuleType,
    name: str,
) -> type[Exception] | None:
    candidate = getattr(module, name, None)
    if isinstance(candidate, type) and issubclass(candidate, Exception):
        return candidate
    return None


def _validate_limits_policy(
    policy: (
        TaskArtifactPolicy | TaskLimitsPolicy | TaskRetryPolicy | TaskRunPolicy
    ),
) -> tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    match policy:
        case TaskRunPolicy():
            _append_limit_issues(
                issues,
                _positive_int_issue(
                    policy.timeout_seconds,
                    "run.timeout_seconds",
                ),
                _non_negative_int_issue(
                    policy.priority,
                    "run.priority",
                ),
                _positive_int_issue(
                    policy.concurrency,
                    "run.concurrency",
                ),
            )
        case TaskRetryPolicy():
            _append_limit_issues(
                issues,
                _positive_int_issue(
                    policy.max_attempts,
                    "retry.max_attempts",
                ),
                _positive_int_issue(
                    policy.max_delay_seconds,
                    "retry.max_delay_seconds",
                ),
            )
        case TaskArtifactPolicy():
            _append_limit_issues(
                issues,
                _positive_int_issue(
                    policy.retention_days,
                    "artifact.retention_days",
                ),
                _positive_int_issue(
                    policy.max_count,
                    "artifact.max_count",
                ),
                _positive_int_issue(
                    policy.max_bytes,
                    "artifact.max_bytes",
                ),
            )
        case TaskLimitsPolicy():
            for field_name in (
                "input_bytes",
                "file_count",
                "file_bytes",
                "output_bytes",
                "artifact_count",
                "artifact_bytes",
                "total_tokens",
            ):
                _append_limit_issues(
                    issues,
                    _positive_int_issue(
                        getattr(policy, field_name),
                        f"limits.{field_name}",
                    ),
                )
    return tuple(issues)


def _append_limit_issues(
    issues: list[TaskValidationIssue],
    *new_issues: TaskValidationIssue | None,
) -> None:
    for issue in new_issues:
        if issue is not None:
            issues.append(issue)


def _positive_int_issue(
    value: object,
    path: str,
) -> TaskValidationIssue | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return None
    return _invalid_limit_issue(path, "Use a positive integer value.")


def _non_negative_int_issue(
    value: object,
    path: str,
) -> TaskValidationIssue | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return None
    return _invalid_limit_issue(path, "Use a non-negative integer value.")


def _is_path_escape(ref: str) -> bool:
    if "://" in ref or "\\" in ref:
        return True
    posix_path = PurePosixPath(ref)
    windows_path = PureWindowsPath(ref)
    if posix_path.is_absolute() or windows_path.is_absolute():
        return True
    return ".." in posix_path.parts or ".." in windows_path.parts


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _schema_type_for_input(input_type: TaskInputType) -> str | None:
    match input_type:
        case TaskInputType.OBJECT:
            return "object"
        case TaskInputType.ARRAY:
            return "array"
        case _:
            return None


def _schema_type_for_output(output_type: TaskOutputType) -> str | None:
    match output_type:
        case TaskOutputType.OBJECT:
            return "object"
        case TaskOutputType.ARRAY:
            return "array"
        case _:
            return None


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and isfinite(value)


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, bool | str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return isfinite(value)
    if isinstance(value, Mapping):
        try:
            return all(
                isinstance(key, str) and _is_json_value(item)
                for key, item in value.items()
            )
        except Exception:
            return False
    if isinstance(value, list | tuple):
        try:
            return all(_is_json_value(item) for item in value)
        except Exception:
            return False
    return False


def _structured_input_schema_value(value: object) -> object:
    if isinstance(value, TaskFileDescriptor):
        descriptor = _file_descriptor_mapping(value)
        assert descriptor is not None
        return {
            key: _structured_input_schema_value(item)
            for key, item in descriptor.items()
        }
    if isinstance(value, TaskFileConversionRequest):
        return {
            "name": value.name,
            "options": _structured_input_schema_value(value.options),
        }
    if isinstance(value, TaskProviderReference):
        return {
            key: _structured_input_schema_value(item)
            for key, item in value.execution_metadata().items()
        }
    if isinstance(value, Mapping):
        try:
            return {
                key: _structured_input_schema_value(item)
                for key, item in value.items()
            }
        except Exception:
            return value
    if isinstance(value, list | tuple):
        try:
            return [_structured_input_schema_value(item) for item in value]
        except Exception:
            return value
    return value


def _json_compatible_data(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible_data(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_json_compatible_data(item) for item in value]
    return value


def _invalid_input_type(
    path: str,
    message: str,
    hint: str,
) -> TaskValidationIssue:
    return _issue(
        code="input.invalid_type",
        path=path,
        message=message,
        hint=hint,
        category=TaskValidationCategory.VALUE,
    )


def _invalid_file_issue(
    path: str,
    message: str,
    hint: str,
) -> TaskValidationIssue:
    return _issue(
        code="input.invalid_file",
        path=path,
        message=message,
        hint=hint,
        category=TaskValidationCategory.VALUE,
    )


def _invalid_output_type(
    path: str,
    message: str,
    hint: str,
) -> TaskValidationIssue:
    return _issue(
        code="output.invalid_type",
        path=path,
        message=message,
        hint=hint,
        category=TaskValidationCategory.VALUE,
    )


def _invalid_schema_issue(
    code: str,
    path: str,
    message: str,
    hint: str,
) -> TaskValidationIssue:
    return _issue(
        code=code,
        path=path,
        message=message,
        hint=hint,
        category=TaskValidationCategory.VALUE,
    )


def _json_schema_missing_issue(path: str) -> TaskValidationIssue:
    return _issue(
        code="dependency.jsonschema_missing",
        path=path,
        message="JSON Schema validation requires the task extra.",
        hint=(
            "Install avalan[task] to validate object, array, or json "
            "contracts."
        ),
        category=TaskValidationCategory.DEPENDENCY,
    )


def _feature_issue(feature: TaskFeature, *, path: str) -> TaskValidationIssue:
    diagnostic = feature_diagnostic(feature, path=path)
    return _issue(
        code=diagnostic.code,
        path=diagnostic.path,
        message=diagnostic.message,
        hint=diagnostic.hint,
        category=TaskValidationCategory.UNSUPPORTED,
    )


def _invalid_limit_issue(path: str, hint: str) -> TaskValidationIssue:
    return _issue(
        code="limits.invalid_value",
        path=path,
        message="Task limit value is invalid.",
        hint=hint,
        category=TaskValidationCategory.VALUE,
    )


def _path_escape_issue() -> TaskValidationIssue:
    return _issue(
        code="execution.path_escape",
        path="execution.ref",
        message="Task execution reference escapes allowed paths.",
        hint=(
            "Use a confined logical reference under an allowed execution root."
        ),
        category=TaskValidationCategory.PRIVACY,
    )


def _issue(
    *,
    code: str,
    path: str,
    message: str,
    hint: str,
    category: TaskValidationCategory,
) -> TaskValidationIssue:
    return TaskValidationIssue(
        code=code,
        path=path,
        message=message,
        hint=hint,
        category=category,
    )
