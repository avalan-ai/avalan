from ..types import (
    assert_bool as _assert_bool,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from .backend import (
    ContainerBackendDiagnostic,
    ContainerBackendDiagnosticCode,
    ContainerBackendStream,
    ContainerBackendStreamChunk,
)
from .conformance import (
    ContainerDiagnostic as ContainerConformanceDiagnostic,
)
from .conformance import (
    ContainerDiagnosticCode,
    ContainerExecutionScope,
)
from .lifecycle import (
    ContainerLifecycleEventStatus,
    ContainerLifecyclePhase,
    ContainerManagedLifecycleResult,
)
from .output import (
    ContainerOutputDiagnostic,
    ContainerOutputDiagnosticCode,
)
from .settings import (
    ContainerAuditEvent,
    ContainerAuditEventType,
    ContainerAuthorizationDecision,
    ContainerAuthorizationDecisionType,
    ContainerExecutionResult,
    ContainerResultStatus,
)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from re import Match
from re import compile as compile_pattern
from types import MappingProxyType
from typing import TypeVar, cast, final

EnumValue = TypeVar("EnumValue", bound=StrEnum)

_SAFE_IDENTIFIER_PATTERN = compile_pattern(
    r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
)
_DIGEST_PATTERN = compile_pattern(r"^sha256:[0-9a-f]{64}$")
_METADATA_KEY_PATTERN = compile_pattern(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_HOST_PATH_PATTERN = compile_pattern(
    r"(?P<path>(?:/Users|/home|/private|/var|/etc|/root)(?:/[^\s,;:\"']*)?)"
)
_MAX_METADATA_VALUE_LENGTH = 240

_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "authorization",
    "credential",
    "password",
    "private",
    "prompt",
    "secret",
    "token",
)
_STREAM_KEY_MARKERS = (
    "content",
    "log",
    "output",
    "stderr",
    "stdout",
)
_STREAM_VALUE_PATTERN = compile_pattern(
    r"(?i)(?:^|[^A-Za-z0-9_])(?:content|log|output|stderr|stdout)"
    r"(?:$|[^A-Za-z0-9_])"
)
_SENSITIVE_VALUE_MARKERS = (
    "authorization:",
    "bearer ",
    "begin private key",
    "password=",
    "private_key",
    "secret=",
    "sk-",
    "token=",
)


class ContainerAuditSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ContainerStableDiagnosticCode(StrEnum):
    AUTO_NOT_ENABLED = "container.backend.auto_not_enabled"
    BACKEND_REQUIRED = "container.backend_required"
    BACKEND_UNAVAILABLE = "container.backend.unavailable"
    CONFORMANCE_BACKEND_UNAVAILABLE = "container.backend_unavailable"
    UNSUPPORTED_SYNTAX = "container.unsupported_syntax"
    CAPABILITY_MISMATCH = "container.backend.capability_mismatch"
    ROOTFUL_NOT_AUTHORIZED = "container.backend.rootful_not_authorized"
    IMAGE_DENIED = "container.backend.image_denied"
    PULL_DENIED = "container.backend.pull_denied"
    PULL_FAILED = "container.backend.pull_failed"
    BUILD_DENIED = "container.backend.build_denied"
    BUILD_FAILED = "container.backend.build_failed"
    CREATE_FAILED = "container.backend.create_failed"
    ATTACH_FAILED = "container.backend.attach_failed"
    START_FAILED = "container.backend.start_failed"
    WAIT_FAILED = "container.backend.wait_failed"
    COPY_FAILED = "container.backend.copy_failed"
    CLEANUP_FAILED = "container.backend.cleanup_failed"
    ORPHAN_QUARANTINED = "container.backend.orphan_quarantined"
    CANCELLED = "container.backend.cancelled"
    TIMEOUT = "container.backend.timeout"
    STREAM_TRUNCATED = "container.backend.stream_truncated"
    EVENT_DROPPED = "container.backend.event_dropped"
    OUTPUT_ABSOLUTE_PATH = "container.output.absolute_path"
    OUTPUT_CASE_COLLISION = "container.output.case_collision"
    OUTPUT_CONTRACT_DISABLED = "container.output.contract_disabled"
    OUTPUT_HARDLINK = "container.output.hardlink"
    OUTPUT_RACE_DETECTED = "container.output.race_detected"
    OUTPUT_PARTIAL_DENIED = "container.output.partial_denied"
    OUTPUT_PARTIAL_QUARANTINED = "container.output.partial_quarantined"
    OUTPUT_SPECIAL_FILE = "container.output.special_file"
    OUTPUT_SYMLINK_ESCAPE = "container.output.symlink_escape"
    OUTPUT_TOO_LARGE = "container.output.too_large"
    OUTPUT_TOO_MANY_FILES = "container.output.too_many_files"
    OUTPUT_TRAVERSAL = "container.output.traversal"
    OUTPUT_UNSAFE_MEDIA = "container.output.unsafe_media"
    OUTPUT_UNSAFE_OWNERSHIP = "container.output.unsafe_ownership"
    OUTPUT_UNSAFE_PERMISSIONS = "container.output.unsafe_permissions"
    OUTPUT_UNSAFE_SIGNATURE = "container.output.unsafe_signature"
    POLICY_DENIED = "container.policy.denied"
    REVIEW_REQUIRED = "container.policy.review_required"
    REVIEW_APPROVAL_STALE = "container.policy.approval_stale"
    UNTRUSTED_AUTHORITY = "container.policy.untrusted_authority"
    UNSAFE_HOST_PATH = "container.security.unsafe_host_path"
    SECRET_LEAK_RISK = "container.security.secret_leak_risk"
    NETWORK_DENIED = "container.security.network_denied"
    DEVICE_DENIED = "container.security.device_denied"
    RESOURCE_DENIED = "container.security.resource_denied"
    ROOT_DENIED = "container.security.root_denied"
    PRIVILEGED_DENIED = "container.security.privileged_denied"
    CAPABILITY_DENIED = "container.security.capability_denied"
    UNKNOWN = "container.unknown_failure"


_BACKEND_CODE_MAP = {
    ContainerBackendDiagnosticCode.AUTO_NOT_ENABLED: (
        ContainerStableDiagnosticCode.AUTO_NOT_ENABLED
    ),
    ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE: (
        ContainerStableDiagnosticCode.BACKEND_UNAVAILABLE
    ),
    ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH: (
        ContainerStableDiagnosticCode.CAPABILITY_MISMATCH
    ),
    ContainerBackendDiagnosticCode.ROOTFUL_NOT_AUTHORIZED: (
        ContainerStableDiagnosticCode.ROOTFUL_NOT_AUTHORIZED
    ),
    ContainerBackendDiagnosticCode.IMAGE_DENIED: (
        ContainerStableDiagnosticCode.IMAGE_DENIED
    ),
    ContainerBackendDiagnosticCode.PULL_DENIED: (
        ContainerStableDiagnosticCode.PULL_DENIED
    ),
    ContainerBackendDiagnosticCode.PULL_FAILED: (
        ContainerStableDiagnosticCode.PULL_FAILED
    ),
    ContainerBackendDiagnosticCode.BUILD_DENIED: (
        ContainerStableDiagnosticCode.BUILD_DENIED
    ),
    ContainerBackendDiagnosticCode.BUILD_FAILED: (
        ContainerStableDiagnosticCode.BUILD_FAILED
    ),
    ContainerBackendDiagnosticCode.CREATE_FAILED: (
        ContainerStableDiagnosticCode.CREATE_FAILED
    ),
    ContainerBackendDiagnosticCode.ATTACH_FAILED: (
        ContainerStableDiagnosticCode.ATTACH_FAILED
    ),
    ContainerBackendDiagnosticCode.START_FAILED: (
        ContainerStableDiagnosticCode.START_FAILED
    ),
    ContainerBackendDiagnosticCode.WAIT_FAILED: (
        ContainerStableDiagnosticCode.WAIT_FAILED
    ),
    ContainerBackendDiagnosticCode.COPY_FAILED: (
        ContainerStableDiagnosticCode.COPY_FAILED
    ),
    ContainerBackendDiagnosticCode.CLEANUP_FAILED: (
        ContainerStableDiagnosticCode.CLEANUP_FAILED
    ),
    ContainerBackendDiagnosticCode.ORPHAN_QUARANTINED: (
        ContainerStableDiagnosticCode.ORPHAN_QUARANTINED
    ),
    ContainerBackendDiagnosticCode.CANCELLED: (
        ContainerStableDiagnosticCode.CANCELLED
    ),
    ContainerBackendDiagnosticCode.TIMEOUT: (
        ContainerStableDiagnosticCode.TIMEOUT
    ),
    ContainerBackendDiagnosticCode.STREAM_TRUNCATED: (
        ContainerStableDiagnosticCode.STREAM_TRUNCATED
    ),
    ContainerBackendDiagnosticCode.EVENT_DROPPED: (
        ContainerStableDiagnosticCode.EVENT_DROPPED
    ),
}
_OUTPUT_CODE_MAP = {
    ContainerOutputDiagnosticCode.ABSOLUTE_PATH: (
        ContainerStableDiagnosticCode.OUTPUT_ABSOLUTE_PATH
    ),
    ContainerOutputDiagnosticCode.CASE_COLLISION: (
        ContainerStableDiagnosticCode.OUTPUT_CASE_COLLISION
    ),
    ContainerOutputDiagnosticCode.CONTRACT_DISABLED: (
        ContainerStableDiagnosticCode.OUTPUT_CONTRACT_DISABLED
    ),
    ContainerOutputDiagnosticCode.HARDLINK: (
        ContainerStableDiagnosticCode.OUTPUT_HARDLINK
    ),
    ContainerOutputDiagnosticCode.RACE_DETECTED: (
        ContainerStableDiagnosticCode.OUTPUT_RACE_DETECTED
    ),
    ContainerOutputDiagnosticCode.PARTIAL_OUTPUT_DENIED: (
        ContainerStableDiagnosticCode.OUTPUT_PARTIAL_DENIED
    ),
    ContainerOutputDiagnosticCode.PARTIAL_OUTPUT_QUARANTINED: (
        ContainerStableDiagnosticCode.OUTPUT_PARTIAL_QUARANTINED
    ),
    ContainerOutputDiagnosticCode.SPECIAL_FILE: (
        ContainerStableDiagnosticCode.OUTPUT_SPECIAL_FILE
    ),
    ContainerOutputDiagnosticCode.SYMLINK_ESCAPE: (
        ContainerStableDiagnosticCode.OUTPUT_SYMLINK_ESCAPE
    ),
    ContainerOutputDiagnosticCode.TOO_LARGE: (
        ContainerStableDiagnosticCode.OUTPUT_TOO_LARGE
    ),
    ContainerOutputDiagnosticCode.TOO_MANY_FILES: (
        ContainerStableDiagnosticCode.OUTPUT_TOO_MANY_FILES
    ),
    ContainerOutputDiagnosticCode.TRAVERSAL: (
        ContainerStableDiagnosticCode.OUTPUT_TRAVERSAL
    ),
    ContainerOutputDiagnosticCode.UNSAFE_MEDIA: (
        ContainerStableDiagnosticCode.OUTPUT_UNSAFE_MEDIA
    ),
    ContainerOutputDiagnosticCode.UNSAFE_OWNERSHIP: (
        ContainerStableDiagnosticCode.OUTPUT_UNSAFE_OWNERSHIP
    ),
    ContainerOutputDiagnosticCode.UNSAFE_PERMISSIONS: (
        ContainerStableDiagnosticCode.OUTPUT_UNSAFE_PERMISSIONS
    ),
    ContainerOutputDiagnosticCode.UNSAFE_SIGNATURE: (
        ContainerStableDiagnosticCode.OUTPUT_UNSAFE_SIGNATURE
    ),
}
_CONFORMANCE_CODE_MAP = {
    ContainerDiagnosticCode.BACKEND_REQUIRED: (
        ContainerStableDiagnosticCode.BACKEND_REQUIRED
    ),
    ContainerDiagnosticCode.BACKEND_UNAVAILABLE: (
        ContainerStableDiagnosticCode.CONFORMANCE_BACKEND_UNAVAILABLE
    ),
    ContainerDiagnosticCode.UNSUPPORTED_SYNTAX: (
        ContainerStableDiagnosticCode.UNSUPPORTED_SYNTAX
    ),
}
_DENIED_CODES = {
    ContainerStableDiagnosticCode.AUTO_NOT_ENABLED,
    ContainerStableDiagnosticCode.BACKEND_REQUIRED,
    ContainerStableDiagnosticCode.CONFORMANCE_BACKEND_UNAVAILABLE,
    ContainerStableDiagnosticCode.UNSUPPORTED_SYNTAX,
    ContainerStableDiagnosticCode.CAPABILITY_MISMATCH,
    ContainerStableDiagnosticCode.ROOTFUL_NOT_AUTHORIZED,
    ContainerStableDiagnosticCode.IMAGE_DENIED,
    ContainerStableDiagnosticCode.PULL_DENIED,
    ContainerStableDiagnosticCode.BUILD_DENIED,
    ContainerStableDiagnosticCode.OUTPUT_ABSOLUTE_PATH,
    ContainerStableDiagnosticCode.OUTPUT_CASE_COLLISION,
    ContainerStableDiagnosticCode.OUTPUT_CONTRACT_DISABLED,
    ContainerStableDiagnosticCode.OUTPUT_HARDLINK,
    ContainerStableDiagnosticCode.OUTPUT_RACE_DETECTED,
    ContainerStableDiagnosticCode.OUTPUT_PARTIAL_DENIED,
    ContainerStableDiagnosticCode.OUTPUT_SPECIAL_FILE,
    ContainerStableDiagnosticCode.OUTPUT_SYMLINK_ESCAPE,
    ContainerStableDiagnosticCode.OUTPUT_TOO_LARGE,
    ContainerStableDiagnosticCode.OUTPUT_TOO_MANY_FILES,
    ContainerStableDiagnosticCode.OUTPUT_TRAVERSAL,
    ContainerStableDiagnosticCode.OUTPUT_UNSAFE_MEDIA,
    ContainerStableDiagnosticCode.OUTPUT_UNSAFE_OWNERSHIP,
    ContainerStableDiagnosticCode.OUTPUT_UNSAFE_PERMISSIONS,
    ContainerStableDiagnosticCode.OUTPUT_UNSAFE_SIGNATURE,
    ContainerStableDiagnosticCode.POLICY_DENIED,
    ContainerStableDiagnosticCode.UNTRUSTED_AUTHORITY,
    ContainerStableDiagnosticCode.UNSAFE_HOST_PATH,
    ContainerStableDiagnosticCode.SECRET_LEAK_RISK,
    ContainerStableDiagnosticCode.NETWORK_DENIED,
    ContainerStableDiagnosticCode.DEVICE_DENIED,
    ContainerStableDiagnosticCode.RESOURCE_DENIED,
    ContainerStableDiagnosticCode.ROOT_DENIED,
    ContainerStableDiagnosticCode.PRIVILEGED_DENIED,
    ContainerStableDiagnosticCode.CAPABILITY_DENIED,
}
_WARNING_CODES = {
    ContainerStableDiagnosticCode.CANCELLED,
    ContainerStableDiagnosticCode.EVENT_DROPPED,
    ContainerStableDiagnosticCode.STREAM_TRUNCATED,
    ContainerStableDiagnosticCode.OUTPUT_PARTIAL_QUARANTINED,
    ContainerStableDiagnosticCode.REVIEW_REQUIRED,
}

_LIFECYCLE_EVENT_TYPES = {
    ContainerLifecyclePhase.POLICY_NORMALIZATION: (
        ContainerAuditEventType.POLICY_EVALUATION
    ),
    ContainerLifecyclePhase.BACKEND_SELECTION: (
        ContainerAuditEventType.BACKEND_SELECTION
    ),
    ContainerLifecyclePhase.IMAGE_RESOLUTION: (
        ContainerAuditEventType.IMAGE_RESOLUTION
    ),
    ContainerLifecyclePhase.IMAGE_PULL: ContainerAuditEventType.IMAGE_PULL,
    ContainerLifecyclePhase.IMAGE_BUILD: (
        ContainerAuditEventType.BUILD_PROGRESS
    ),
    ContainerLifecyclePhase.CREATE: ContainerAuditEventType.CONTAINER_CREATE,
    ContainerLifecyclePhase.START: ContainerAuditEventType.CONTAINER_START,
    ContainerLifecyclePhase.STATS: ContainerAuditEventType.STATS,
    ContainerLifecyclePhase.WAIT: ContainerAuditEventType.EXIT,
    ContainerLifecyclePhase.COPY_OUTPUTS: ContainerAuditEventType.OUTPUT_COPY,
    ContainerLifecyclePhase.REMOVE: ContainerAuditEventType.CLEANUP,
    ContainerLifecyclePhase.CLEANUP: ContainerAuditEventType.CLEANUP,
}


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerAuditCorrelation:
    profile_name: str | None
    policy_version: str
    agent_id: str | None = None
    session_id: str | None = None
    tool_call_id: str | None = None
    flow_node_id: str | None = None
    task_run_id: str | None = None
    attempt_id: str | None = None
    image_digest: str | None = None

    def __post_init__(self) -> None:
        if self.profile_name is not None:
            _assert_safe_identifier(self.profile_name, "profile_name")
        _assert_safe_identifier(self.policy_version, "policy_version")
        for field_name in (
            "agent_id",
            "session_id",
            "tool_call_id",
            "flow_node_id",
            "task_run_id",
            "attempt_id",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _assert_safe_identifier(value, field_name)
        if self.image_digest is not None:
            assert _DIGEST_PATTERN.match(
                self.image_digest
            ), "image_digest must be a sha256 digest"

    def to_metadata(self) -> dict[str, str]:
        metadata = {"policy_version": self.policy_version}
        for field_name in (
            "agent_id",
            "session_id",
            "tool_call_id",
            "flow_node_id",
            "task_run_id",
            "attempt_id",
            "profile_name",
            "image_digest",
        ):
            value = getattr(self, field_name)
            if value is not None:
                metadata[field_name] = value
        return metadata


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerMappedDiagnostic:
    code: ContainerStableDiagnosticCode | str
    message: str
    status: ContainerResultStatus | str
    source_code: str | None = None
    retryable: bool = False
    severity: ContainerAuditSeverity | str = ContainerAuditSeverity.ERROR
    event_type: ContainerAuditEventType | str = ContainerAuditEventType.FAILURE

    def __post_init__(self) -> None:
        code = _enum_value(
            self.code,
            ContainerStableDiagnosticCode,
            "code",
        )
        object.__setattr__(self, "code", code)
        object.__setattr__(
            self,
            "status",
            _enum_value(self.status, ContainerResultStatus, "status"),
        )
        object.__setattr__(
            self,
            "severity",
            _enum_value(self.severity, ContainerAuditSeverity, "severity"),
        )
        object.__setattr__(
            self,
            "event_type",
            _enum_value(
                self.event_type,
                ContainerAuditEventType,
                "event_type",
            ),
        )
        object.__setattr__(
            self,
            "message",
            redact_container_audit_value("diagnostic_message", self.message),
        )
        if self.source_code is not None:
            _assert_non_empty_string(self.source_code, "source_code")
        _assert_bool(self.retryable, "retryable")

    def to_dict(self) -> dict[str, object]:
        code = cast(ContainerStableDiagnosticCode, self.code)
        status = cast(ContainerResultStatus, self.status)
        severity = cast(ContainerAuditSeverity, self.severity)
        event_type = cast(ContainerAuditEventType, self.event_type)
        return {
            "code": code.value,
            "source_code": self.source_code,
            "status": status.value,
            "severity": severity.value,
            "event_type": event_type.value,
            "message": self.message,
            "retryable": self.retryable,
        }

    def result_message(self) -> str:
        code = cast(ContainerStableDiagnosticCode, self.code)
        return f"{code.value}: {self.message}"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerAuditRecord:
    event_type: ContainerAuditEventType | str
    scope: ContainerExecutionScope | str
    correlation: ContainerAuditCorrelation
    metadata: Mapping[str, object] = field(default_factory=dict)
    diagnostics: Sequence[ContainerMappedDiagnostic] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_type",
            _enum_value(
                self.event_type,
                ContainerAuditEventType,
                "event_type",
            ),
        )
        object.__setattr__(
            self,
            "scope",
            _enum_value(self.scope, ContainerExecutionScope, "scope"),
        )
        assert isinstance(self.correlation, ContainerAuditCorrelation)
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, ContainerMappedDiagnostic)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(sanitize_container_audit_metadata(self.metadata)),
        )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def to_event(self) -> ContainerAuditEvent:
        metadata = self.correlation.to_metadata() | {
            key: str(value) for key, value in self.metadata.items()
        }
        if self.diagnostics:
            metadata["diagnostic_codes"] = ",".join(
                cast(ContainerStableDiagnosticCode, diagnostic.code).value
                for diagnostic in self.diagnostics
            )
        return ContainerAuditEvent(
            event_type=cast(ContainerAuditEventType, self.event_type),
            scope=cast(ContainerExecutionScope, self.scope),
            profile_name=self.correlation.profile_name,
            policy_version=self.correlation.policy_version,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, object]:
        event_type = cast(ContainerAuditEventType, self.event_type)
        scope = cast(ContainerExecutionScope, self.scope)
        return {
            "event_type": event_type.value,
            "scope": scope.value,
            "correlation": self.correlation.to_metadata(),
            "metadata": dict(self.metadata),
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerFormattedOutput:
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.text, "text")
        object.__setattr__(
            self,
            "text",
            redact_container_audit_value("text", self.text),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(sanitize_container_audit_metadata(self.metadata)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "metadata": dict(self.metadata),
        }


def sanitize_container_audit_metadata(
    metadata: Mapping[str, object],
) -> dict[str, str]:
    assert isinstance(metadata, Mapping), "metadata must be a mapping"
    sanitized: dict[str, str] = {}
    for key, value in metadata.items():
        assert isinstance(key, str), "metadata keys must be strings"
        assert _METADATA_KEY_PATTERN.match(
            key
        ), "metadata keys must be safe identifiers"
        sanitized[key] = redact_container_audit_value(key, value)
    return sanitized


def redact_container_audit_value(key: str, value: object) -> str:
    _assert_non_empty_string(key, "key")
    lowered_key = key.lower()
    if _contains_marker(lowered_key, _STREAM_KEY_MARKERS):
        return "<redacted-stream>"
    if isinstance(value, bytes):
        return "<redacted-bytes>"
    text = str(value)
    if lowered_key == "diagnostic_message" and _STREAM_VALUE_PATTERN.search(
        text
    ):
        return "<redacted-stream>"
    lowered_text = text.lower()
    if _contains_marker(lowered_key, _SENSITIVE_KEY_MARKERS):
        return "<redacted>"
    if _contains_marker(lowered_text, _SENSITIVE_VALUE_MARKERS):
        return "<redacted>"
    if _contains_nonprintable(text):
        return "<redacted-bytes>"
    redacted = _HOST_PATH_PATTERN.sub(_redact_host_path_match, text)
    if len(redacted) > _MAX_METADATA_VALUE_LENGTH:
        return f"{redacted[:_MAX_METADATA_VALUE_LENGTH]}...<truncated>"
    return redacted


def normalize_container_diagnostic(
    diagnostic: object,
) -> ContainerMappedDiagnostic:
    if isinstance(diagnostic, ContainerMappedDiagnostic):
        return diagnostic
    if isinstance(diagnostic, ContainerStableDiagnosticCode):
        return _mapped_diagnostic(diagnostic, diagnostic.value)
    if isinstance(diagnostic, ContainerBackendDiagnostic):
        backend_code = cast(ContainerBackendDiagnosticCode, diagnostic.code)
        code = _BACKEND_CODE_MAP[backend_code]
        return _mapped_diagnostic(
            code,
            diagnostic.message,
            source_code=backend_code.value,
            retryable=diagnostic.retryable,
        )
    if isinstance(diagnostic, ContainerOutputDiagnostic):
        output_code = cast(ContainerOutputDiagnosticCode, diagnostic.code)
        code = _OUTPUT_CODE_MAP[output_code]
        return _mapped_diagnostic(
            code,
            diagnostic.message,
            source_code=output_code.value,
        )
    if isinstance(diagnostic, ContainerConformanceDiagnostic):
        code = _CONFORMANCE_CODE_MAP[diagnostic.code]
        return _mapped_diagnostic(
            code,
            diagnostic.message,
            source_code=diagnostic.code.value,
        )
    if isinstance(diagnostic, ContainerAuthorizationDecision):
        return _mapped_authorization_decision(diagnostic)
    if isinstance(diagnostic, str):
        return _mapped_diagnostic(
            ContainerStableDiagnosticCode.UNKNOWN,
            diagnostic,
        )
    raise AssertionError("unsupported diagnostic type")


def container_execution_result_from_diagnostics(
    diagnostics: Sequence[object],
    correlation: ContainerAuditCorrelation,
    *,
    exit_code: int | None = None,
) -> ContainerExecutionResult:
    assert isinstance(correlation, ContainerAuditCorrelation)
    mapped = tuple(
        normalize_container_diagnostic(item) for item in diagnostics
    )
    status = _status_from_mapped_diagnostics(mapped)
    metadata = correlation.to_metadata() | {
        "diagnostic_count": str(len(mapped)),
        "diagnostic_codes": _diagnostic_codes_csv(mapped),
    }
    return ContainerExecutionResult(
        status=status,
        exit_code=exit_code,
        diagnostics=tuple(
            diagnostic.result_message() for diagnostic in mapped
        ),
        metadata=metadata,
    )


def format_container_diagnostics_for_model(
    diagnostics: Sequence[object],
    correlation: ContainerAuditCorrelation,
) -> ContainerFormattedOutput:
    assert isinstance(correlation, ContainerAuditCorrelation)
    mapped = tuple(
        normalize_container_diagnostic(item) for item in diagnostics
    )
    status = _status_from_mapped_diagnostics(mapped)
    if not mapped:
        text = f"container status: {status.value}"
    else:
        lines = [f"container status: {status.value}", "diagnostics:"]
        lines.extend(
            f"- {cast(ContainerStableDiagnosticCode, item.code).value}: "
            f"{item.message}"
            for item in mapped
        )
        text = "\n".join(lines)
    metadata = correlation.to_metadata() | {
        "status": status.value,
        "diagnostic_count": str(len(mapped)),
        "diagnostic_codes": _diagnostic_codes_csv(mapped),
    }
    return ContainerFormattedOutput(text=text, metadata=metadata)


def container_diagnostic_audit_event(
    diagnostic: object,
    correlation: ContainerAuditCorrelation,
    *,
    scope: ContainerExecutionScope | str,
) -> ContainerAuditRecord:
    mapped = normalize_container_diagnostic(diagnostic)
    code = cast(ContainerStableDiagnosticCode, mapped.code)
    status = cast(ContainerResultStatus, mapped.status)
    event_type = cast(ContainerAuditEventType, mapped.event_type)
    return ContainerAuditRecord(
        event_type=event_type,
        scope=scope,
        correlation=correlation,
        diagnostics=(mapped,),
        metadata={
            "code": code.value,
            "status": status.value,
            "retryable": str(mapped.retryable).lower(),
        },
    )


def container_lifecycle_audit_events(
    result: ContainerManagedLifecycleResult,
    correlation: ContainerAuditCorrelation,
    *,
    scope: ContainerExecutionScope | str,
    include_review: bool = False,
    include_mount_preparation: bool = True,
) -> tuple[ContainerAuditRecord, ...]:
    assert isinstance(result, ContainerManagedLifecycleResult)
    assert isinstance(correlation, ContainerAuditCorrelation)
    records: list[ContainerAuditRecord] = []
    if include_review:
        records.append(
            _audit_record(
                ContainerAuditEventType.REVIEW_REQUEST,
                scope,
                correlation,
            )
        )
        records.append(
            _audit_record(
                ContainerAuditEventType.REVIEW_DECISION,
                scope,
                correlation,
                {"decision": "allow"},
            )
        )
    if include_mount_preparation:
        records.append(
            _audit_record(
                ContainerAuditEventType.MOUNT_PREPARATION,
                scope,
                correlation,
            )
        )
    for event in result.events:
        if event.status is not ContainerLifecycleEventStatus.STARTED:
            continue
        event_type = _LIFECYCLE_EVENT_TYPES.get(
            cast(ContainerLifecyclePhase, event.phase)
        )
        if event_type is None:
            continue
        records.append(
            _audit_record(
                event_type,
                scope,
                correlation,
                {
                    "phase": cast(ContainerLifecyclePhase, event.phase).value,
                    "phase_status": event.status.value,
                },
            )
        )
    records.extend(
        _stream_audit_record(chunk, correlation, scope)
        for chunk in result.stream.chunks
    )
    if result.timed_out_phase is not None:
        records.append(
            _audit_record(
                ContainerAuditEventType.TIMEOUT,
                scope,
                correlation,
                {
                    "phase": (
                        cast(
                            ContainerLifecyclePhase,
                            result.timed_out_phase,
                        ).value
                    )
                },
            )
        )
    if result.cancelled_phase is not None:
        records.append(
            _audit_record(
                ContainerAuditEventType.CANCELLATION,
                scope,
                correlation,
                {
                    "phase": (
                        cast(
                            ContainerLifecyclePhase,
                            result.cancelled_phase,
                        ).value
                    )
                },
            )
        )
    records.extend(
        container_diagnostic_audit_event(
            diagnostic,
            correlation,
            scope=scope,
        )
        for diagnostic in result.diagnostics
    )
    status = cast(ContainerResultStatus, result.execution.status)
    if status is ContainerResultStatus.DENIED:
        records.append(
            _audit_record(ContainerAuditEventType.DENIAL, scope, correlation)
        )
    if status is ContainerResultStatus.FAILED:
        records.append(
            _audit_record(ContainerAuditEventType.FAILURE, scope, correlation)
        )
    records.append(
        _audit_record(
            ContainerAuditEventType.RESULT_RECORDED,
            scope,
            correlation,
            {"status": status.value},
        )
    )
    return tuple(records)


def _mapped_diagnostic(
    code: ContainerStableDiagnosticCode,
    message: str,
    *,
    source_code: str | None = None,
    retryable: bool = False,
) -> ContainerMappedDiagnostic:
    status = _status_for_stable_code(code)
    return ContainerMappedDiagnostic(
        code=code,
        message=message,
        status=status,
        source_code=source_code,
        retryable=retryable,
        severity=_severity_for_code(code),
        event_type=_event_type_for_code(code, status),
    )


def _mapped_authorization_decision(
    decision: ContainerAuthorizationDecision,
) -> ContainerMappedDiagnostic:
    decision_type = cast(ContainerAuthorizationDecisionType, decision.decision)
    if decision_type is ContainerAuthorizationDecisionType.REQUIRES_REVIEW:
        code = ContainerStableDiagnosticCode.REVIEW_REQUIRED
    elif decision_type is ContainerAuthorizationDecisionType.DENY:
        code = ContainerStableDiagnosticCode.POLICY_DENIED
    else:
        code = ContainerStableDiagnosticCode.UNKNOWN
    return _mapped_diagnostic(
        code,
        decision.explanation,
        source_code=decision.code,
        retryable=decision.retryable,
    )


def _status_for_stable_code(
    code: ContainerStableDiagnosticCode,
) -> ContainerResultStatus:
    if code is ContainerStableDiagnosticCode.CANCELLED:
        return ContainerResultStatus.CANCELLED
    if code in _DENIED_CODES:
        return ContainerResultStatus.DENIED
    return ContainerResultStatus.FAILED


def _severity_for_code(
    code: ContainerStableDiagnosticCode,
) -> ContainerAuditSeverity:
    if code in _WARNING_CODES:
        return ContainerAuditSeverity.WARNING
    return ContainerAuditSeverity.ERROR


def _event_type_for_code(
    code: ContainerStableDiagnosticCode,
    status: ContainerResultStatus,
) -> ContainerAuditEventType:
    if code is ContainerStableDiagnosticCode.TIMEOUT:
        return ContainerAuditEventType.TIMEOUT
    if status is ContainerResultStatus.CANCELLED:
        return ContainerAuditEventType.CANCELLATION
    if status is ContainerResultStatus.DENIED:
        return ContainerAuditEventType.DENIAL
    return ContainerAuditEventType.FAILURE


def _status_from_mapped_diagnostics(
    diagnostics: Sequence[ContainerMappedDiagnostic],
) -> ContainerResultStatus:
    if not diagnostics:
        return ContainerResultStatus.COMPLETED
    statuses = {
        cast(ContainerResultStatus, item.status) for item in diagnostics
    }
    if ContainerResultStatus.CANCELLED in statuses:
        return ContainerResultStatus.CANCELLED
    if ContainerResultStatus.DENIED in statuses:
        return ContainerResultStatus.DENIED
    return ContainerResultStatus.FAILED


def _diagnostic_codes_csv(
    diagnostics: Sequence[ContainerMappedDiagnostic],
) -> str:
    if not diagnostics:
        return "none"
    return ",".join(
        cast(ContainerStableDiagnosticCode, diagnostic.code).value
        for diagnostic in diagnostics
    )


def _audit_record(
    event_type: ContainerAuditEventType,
    scope: ContainerExecutionScope | str,
    correlation: ContainerAuditCorrelation,
    metadata: Mapping[str, object] | None = None,
) -> ContainerAuditRecord:
    return ContainerAuditRecord(
        event_type=event_type,
        scope=scope,
        correlation=correlation,
        metadata=metadata or {},
    )


def _stream_audit_record(
    chunk: ContainerBackendStreamChunk,
    correlation: ContainerAuditCorrelation,
    scope: ContainerExecutionScope | str,
) -> ContainerAuditRecord:
    stream = cast(ContainerBackendStream, chunk.stream)
    event_type = {
        ContainerBackendStream.STDOUT: ContainerAuditEventType.STDOUT_CHUNK,
        ContainerBackendStream.STDERR: ContainerAuditEventType.STDERR_CHUNK,
        ContainerBackendStream.PROGRESS: ContainerAuditEventType.PROGRESS,
    }[stream]
    return _audit_record(
        event_type,
        scope,
        correlation,
        {
            "stream": stream.value,
            "sequence": str(chunk.sequence),
            "byte_count": str(len(chunk.content)),
            "content": chunk.content,
        },
    )


def _contains_marker(value: str, markers: Sequence[str]) -> bool:
    return any(marker in value for marker in markers)


def _contains_nonprintable(value: str) -> bool:
    return any(
        not character.isprintable() and character not in {"\n", "\r", "\t"}
        for character in value
    )


def _redact_host_path_match(match: Match[str]) -> str:
    path = match.group("path")
    name = Path(path).name
    return f"<host-path>/{name}" if name else "<host-path>"


def _assert_safe_identifier(value: str, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert _SAFE_IDENTIFIER_PATTERN.match(
        value
    ), f"{field_name} must be a safe identifier"


def _enum_value(
    value: object,
    enum_type: type[EnumValue],
    field_name: str,
) -> EnumValue:
    if isinstance(value, enum_type):
        return value
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert value in {
        member.value for member in enum_type
    }, f"{field_name} contains unsupported value"
    return enum_type(value)
