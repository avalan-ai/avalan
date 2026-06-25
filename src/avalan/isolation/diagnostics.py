from ..types import (
    assert_bool as _assert_bool,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from .settings import (
    IsolationDiagnostic,
    IsolationDiagnosticCategory,
    IsolationDiagnosticCode,
    IsolationDiagnosticSeverity,
)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from re import Match
from re import compile as compile_pattern
from types import MappingProxyType
from typing import TypeVar, cast, final

EnumValue = TypeVar("EnumValue", bound=StrEnum)

STABLE_ISOLATION_DIAGNOSTIC_CODES = (
    IsolationDiagnosticCode.MODE_CONFLICT,
    IsolationDiagnosticCode.UNSUPPORTED_MODE,
    IsolationDiagnosticCode.UNSUPPORTED_BACKEND,
    IsolationDiagnosticCode.MODE_UNAVAILABLE,
    IsolationDiagnosticCode.CAPABILITY_MISMATCH,
    IsolationDiagnosticCode.ELEVATION_REQUIRED,
    IsolationDiagnosticCode.ELEVATION_DENIED,
    IsolationDiagnosticCode.FALLBACK_DENIED,
    IsolationDiagnosticCode.APPROVAL_STALE,
    IsolationDiagnosticCode.POLICY_DRIFT,
    IsolationDiagnosticCode.AUDIT_UNAVAILABLE,
    IsolationDiagnosticCode.SANDBOX_PROVIDER_UNAVAILABLE,
    IsolationDiagnosticCode.SANDBOX_PROFILE_GENERATION_FAILED,
    IsolationDiagnosticCode.SANDBOX_PATH_DENIED,
    IsolationDiagnosticCode.SANDBOX_NETWORK_UNENFORCEABLE,
    IsolationDiagnosticCode.CONTAINER_BACKEND_UNAVAILABLE,
    IsolationDiagnosticCode.CONTAINER_BACKEND_CAPABILITY_MISMATCH,
)

_STABLE_CODE_SET = frozenset(STABLE_ISOLATION_DIAGNOSTIC_CODES)
_STABLE_CODE_VALUES = frozenset(
    code.value for code in STABLE_ISOLATION_DIAGNOSTIC_CODES
)
_METADATA_KEY_PATTERN = compile_pattern(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_HOST_PATH_PATTERN = compile_pattern(
    r"(?P<path>(?:(?:/Users|/home|/private|/var|/etc|/root)"
    r"|(?:/tmp|/opt|/srv|/mnt|/Volumes))"
    r"(?:/[^\s,;:\"']*)?)"
)
_MAX_METADATA_VALUE_LENGTH = 240
_MAX_DIAGNOSTIC_METADATA_VALUE_LENGTH = 2048
_MAX_FORMATTED_DIAGNOSTIC_TEXT_LENGTH = 4096
_DIAGNOSTIC_METADATA_KEYS = (
    "diagnostic_audit_events",
    "diagnostic_codes",
    "diagnostic_source_codes",
)
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
_STREAM_VALUE_PATTERN = compile_pattern(
    r"(?i)(?:^|[^A-Za-z0-9_])(?:content|log|output|stderr|stdout)"
    r"(?:$|[^A-Za-z0-9_])"
)


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationDiagnosticInventoryItem:
    code: IsolationDiagnosticCode | str
    category: IsolationDiagnosticCategory | str
    severity: IsolationDiagnosticSeverity | str
    message: str
    hint: str
    audit_event: str
    model_status: str
    metadata_fields: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        code = _stable_code(self.code)
        category = _enum_value(
            self.category,
            IsolationDiagnosticCategory,
            "category",
        )
        severity = _enum_value(
            self.severity,
            IsolationDiagnosticSeverity,
            "severity",
        )
        _assert_non_empty_string(self.message, "message")
        _assert_non_empty_string(self.hint, "hint")
        _assert_non_empty_string(self.audit_event, "audit_event")
        _assert_non_empty_string(self.model_status, "model_status")
        metadata_fields = _metadata_field_tuple(self.metadata_fields)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "metadata_fields", metadata_fields)

    def to_dict(self) -> dict[str, object]:
        code = cast(IsolationDiagnosticCode, self.code)
        category = cast(IsolationDiagnosticCategory, self.category)
        severity = cast(IsolationDiagnosticSeverity, self.severity)
        return {
            "code": code.value,
            "category": category.value,
            "severity": severity.value,
            "message": self.message,
            "hint": self.hint,
            "audit_event": self.audit_event,
            "model_status": self.model_status,
            "metadata_fields": list(self.metadata_fields),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationMappedDiagnostic:
    code: IsolationDiagnosticCode | str
    message: str
    hint: str
    category: IsolationDiagnosticCategory | str
    severity: IsolationDiagnosticSeverity | str
    audit_event: str
    model_status: str
    path: str = "runtime.isolation"
    source_code: str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        code = _stable_code(self.code)
        category = _enum_value(
            self.category,
            IsolationDiagnosticCategory,
            "category",
        )
        severity = _enum_value(
            self.severity,
            IsolationDiagnosticSeverity,
            "severity",
        )
        _assert_non_empty_string(self.path, "path")
        _assert_non_empty_string(self.message, "message")
        _assert_non_empty_string(self.hint, "hint")
        _assert_non_empty_string(self.audit_event, "audit_event")
        _assert_non_empty_string(self.model_status, "model_status")
        if self.source_code is not None:
            _assert_non_empty_string(self.source_code, "source_code")
        _assert_bool(self.retryable, "retryable")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(
            self,
            "message",
            redact_isolation_value("diagnostic_message", self.message),
        )
        object.__setattr__(
            self,
            "hint",
            redact_isolation_value("diagnostic_hint", self.hint),
        )
        object.__setattr__(
            self,
            "path",
            redact_isolation_value("path", self.path),
        )

    def to_public_dict(self) -> dict[str, str]:
        code = cast(IsolationDiagnosticCode, self.code)
        category = cast(IsolationDiagnosticCategory, self.category)
        severity = cast(IsolationDiagnosticSeverity, self.severity)
        return {
            "code": code.value,
            "path": self.path,
            "category": category.value,
            "severity": severity.value,
            "message": self.message,
            "hint": self.hint,
        }

    def to_metadata(self) -> dict[str, str]:
        code = cast(IsolationDiagnosticCode, self.code)
        category = cast(IsolationDiagnosticCategory, self.category)
        severity = cast(IsolationDiagnosticSeverity, self.severity)
        metadata = {
            "code": code.value,
            "category": category.value,
            "severity": severity.value,
            "audit_event": self.audit_event,
            "model_status": self.model_status,
            "retryable": str(self.retryable).lower(),
        }
        if self.source_code is not None:
            metadata["source_code"] = self.source_code
        return sanitize_isolation_metadata(metadata)


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationFormattedOutput:
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.text, "text")
        object.__setattr__(
            self,
            "text",
            redact_isolation_value("text", self.text),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(sanitize_isolation_metadata(self.metadata)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "metadata": dict(self.metadata),
        }


def stable_isolation_diagnostic_inventory() -> (
    tuple[IsolationDiagnosticInventoryItem, ...]
):
    return _INVENTORY_ITEMS


def stable_isolation_diagnostic_codes() -> tuple[str, ...]:
    return tuple(code.value for code in STABLE_ISOLATION_DIAGNOSTIC_CODES)


def normalize_isolation_diagnostic(
    diagnostic: object,
) -> IsolationMappedDiagnostic:
    if isinstance(diagnostic, IsolationMappedDiagnostic):
        return diagnostic
    source_code = _diagnostic_source_code(diagnostic)
    stable_code = _stable_code_for_source(
        source_code,
        operation=_diagnostic_operation(diagnostic),
        message=_diagnostic_message(diagnostic),
    )
    item = _INVENTORY_BY_CODE[stable_code]
    source = None if source_code == stable_code.value else source_code
    return IsolationMappedDiagnostic(
        code=stable_code,
        path=_diagnostic_path(diagnostic),
        message=_diagnostic_message(diagnostic) or item.message,
        hint=_diagnostic_hint(diagnostic) or item.hint,
        category=_diagnostic_category(diagnostic) or item.category,
        severity=item.severity,
        audit_event=item.audit_event,
        model_status=item.model_status,
        source_code=source,
        retryable=_diagnostic_retryable(diagnostic),
    )


def isolation_diagnostic_codes(
    diagnostics: Sequence[object],
) -> tuple[str, ...]:
    codes: list[str] = []
    for diagnostic in diagnostics:
        try:
            mapped = normalize_isolation_diagnostic(diagnostic)
        except AssertionError:
            continue
        code = cast(IsolationDiagnosticCode, mapped.code).value
        if code not in codes:
            codes.append(code)
    return tuple(codes)


def isolation_public_diagnostics(
    diagnostics: Sequence[object],
) -> tuple[dict[str, str], ...]:
    return tuple(
        normalize_isolation_diagnostic(diagnostic).to_public_dict()
        for diagnostic in diagnostics
    )


def isolation_diagnostics_metadata(
    diagnostics: Sequence[object],
    metadata: Mapping[str, object] | None = None,
) -> dict[str, str]:
    mapped = tuple(
        normalize_isolation_diagnostic(diagnostic)
        for diagnostic in diagnostics
    )
    diagnostic_codes = tuple(
        cast(IsolationDiagnosticCode, diagnostic.code).value
        for diagnostic in mapped
    )
    source_codes = tuple(
        diagnostic.source_code
        for diagnostic in mapped
        if diagnostic.source_code is not None
    )
    audit_events = tuple(
        dict.fromkeys(diagnostic.audit_event for diagnostic in mapped)
    )
    values: dict[str, object] = dict(metadata or {})
    values["diagnostic_count"] = str(len(mapped))
    values["diagnostic_codes"] = ",".join(diagnostic_codes)
    if source_codes:
        values["diagnostic_source_codes"] = ",".join(source_codes)
    if audit_events:
        values["diagnostic_audit_events"] = ",".join(audit_events)
    return sanitize_isolation_metadata(values)


def isolation_diagnostic_audit_metadata(
    diagnostic: object,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, str]:
    mapped = normalize_isolation_diagnostic(diagnostic)
    values: dict[str, object] = dict(metadata or {})
    values.update(mapped.to_metadata())
    return isolation_diagnostics_metadata((mapped,), values)


def format_isolation_diagnostics_for_model(
    diagnostics: Sequence[object],
    metadata: Mapping[str, object] | None = None,
) -> str:
    if metadata is not None:
        sanitize_isolation_metadata(metadata)
    mapped = tuple(
        normalize_isolation_diagnostic(diagnostic)
        for diagnostic in diagnostics
    )
    if not mapped:
        return "isolation: ok"
    return "\n".join(
        f"{cast(IsolationDiagnosticCode, diagnostic.code).value}: "
        f"{diagnostic.message} {diagnostic.hint}"
        for diagnostic in mapped
    )


def format_isolation_diagnostics_output_for_model(
    diagnostics: Sequence[object],
    metadata: Mapping[str, object] | None = None,
) -> IsolationFormattedOutput:
    mapped = tuple(
        normalize_isolation_diagnostic(diagnostic)
        for diagnostic in diagnostics
    )
    if not mapped:
        text = "isolation status: ok"
    else:
        status = _combined_model_status(mapped)
        lines = [f"isolation status: {status}", "diagnostics:"]
        lines.extend(
            f"- {cast(IsolationDiagnosticCode, diagnostic.code).value}: "
            f"{diagnostic.message} {diagnostic.hint}"
            for diagnostic in mapped
        )
        text = "\n".join(lines)
    return IsolationFormattedOutput(
        text=text,
        metadata=isolation_diagnostics_metadata(mapped, metadata),
    )


def sanitize_isolation_metadata(
    metadata: Mapping[str, object],
) -> dict[str, str]:
    assert isinstance(metadata, Mapping), "metadata must be a mapping"
    sanitized: dict[str, str] = {}
    for key, value in metadata.items():
        assert isinstance(key, str), "metadata keys must be strings"
        assert _METADATA_KEY_PATTERN.match(
            key
        ), "metadata keys must be safe identifiers"
        sanitized[key] = redact_isolation_value(key, value)
    return sanitized


def redact_isolation_value(key: str, value: object) -> str:
    _assert_non_empty_string(key, "key")
    lowered_key = key.lower()
    if _contains_marker(lowered_key, _STREAM_KEY_MARKERS):
        return "<redacted-stream>"
    if isinstance(value, bytes):
        return "<redacted-bytes>"
    text = str(value)
    if lowered_key in {"diagnostic_message", "diagnostic_hint", "text"}:
        if _STREAM_VALUE_PATTERN.search(text):
            return "<redacted-stream>"
    lowered_text = text.lower()
    if _contains_marker(lowered_key, _SENSITIVE_KEY_MARKERS):
        return "<redacted>"
    if _contains_marker(lowered_text, _SENSITIVE_VALUE_MARKERS):
        return "<redacted>"
    if _contains_nonprintable(text):
        return "<redacted-bytes>"
    redacted = _HOST_PATH_PATTERN.sub(_redact_host_path_match, text)
    max_length = (
        _MAX_FORMATTED_DIAGNOSTIC_TEXT_LENGTH
        if lowered_key == "text"
        else (
            _MAX_DIAGNOSTIC_METADATA_VALUE_LENGTH
            if lowered_key in _DIAGNOSTIC_METADATA_KEYS
            else _MAX_METADATA_VALUE_LENGTH
        )
    )
    if len(redacted) > max_length:
        return f"{redacted[:max_length]}...<truncated>"
    return redacted


def _inventory_item(
    code: IsolationDiagnosticCode,
    *,
    category: IsolationDiagnosticCategory,
    severity: IsolationDiagnosticSeverity,
    message: str,
    hint: str,
    audit_event: str,
    model_status: str,
) -> IsolationDiagnosticInventoryItem:
    return IsolationDiagnosticInventoryItem(
        code=code,
        category=category,
        severity=severity,
        message=message,
        hint=hint,
        audit_event=audit_event,
        model_status=model_status,
        metadata_fields=(
            "diagnostic_count",
            "diagnostic_codes",
            "diagnostic_source_codes",
            "diagnostic_audit_events",
        ),
    )


def _stable_code(
    value: IsolationDiagnosticCode | str,
) -> IsolationDiagnosticCode:
    code = _enum_value(value, IsolationDiagnosticCode, "code")
    assert code in _STABLE_CODE_SET, "code is not in the stable inventory"
    return code


def _enum_value(
    value: EnumValue | str,
    enum_type: type[EnumValue],
    field_name: str,
) -> EnumValue:
    if isinstance(value, enum_type):
        return value
    assert isinstance(value, str), f"{field_name} must be a string"
    try:
        return enum_type(value)
    except ValueError as exc:
        raise AssertionError(f"unsupported {field_name}: {value}") from exc


def _metadata_field_tuple(value: object) -> tuple[str, ...]:
    assert isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes,
    ), "metadata_fields must be a sequence"
    fields: list[str] = []
    for item in value:
        assert isinstance(item, str), "metadata_fields must contain strings"
        assert _METADATA_KEY_PATTERN.match(
            item
        ), "metadata_fields must be safe identifiers"
        fields.append(item)
    return tuple(fields)


def _diagnostic_source_code(diagnostic: object) -> str:
    if isinstance(diagnostic, IsolationDiagnosticCode):
        return diagnostic.value
    if isinstance(diagnostic, str):
        _assert_non_empty_string(diagnostic, "diagnostic")
        return diagnostic
    if isinstance(diagnostic, IsolationDiagnostic):
        code = cast(IsolationDiagnosticCode, diagnostic.code)
        return code.value
    source_code = getattr(diagnostic, "code", None)
    if isinstance(source_code, StrEnum):
        return source_code.value
    if isinstance(source_code, str):
        _assert_non_empty_string(source_code, "code")
        return source_code
    raise AssertionError("unsupported diagnostic type")


def _diagnostic_operation(diagnostic: object) -> str | None:
    operation = getattr(diagnostic, "operation", None)
    if isinstance(operation, StrEnum):
        return operation.value
    if isinstance(operation, str):
        return operation
    return None


def _diagnostic_message(diagnostic: object) -> str:
    message = getattr(diagnostic, "message", "")
    return message if isinstance(message, str) else ""


def _diagnostic_hint(diagnostic: object) -> str:
    hint = getattr(diagnostic, "hint", "")
    return hint if isinstance(hint, str) else ""


def _diagnostic_path(diagnostic: object) -> str:
    path = getattr(diagnostic, "path", "runtime.isolation")
    return path if isinstance(path, str) and path else "runtime.isolation"


def _diagnostic_category(
    diagnostic: object,
) -> IsolationDiagnosticCategory | None:
    if not isinstance(diagnostic, IsolationDiagnostic):
        return None
    return cast(IsolationDiagnosticCategory, diagnostic.category)


def _diagnostic_retryable(diagnostic: object) -> bool:
    retryable = getattr(diagnostic, "retryable", False)
    return retryable if isinstance(retryable, bool) else False


def _stable_code_for_source(
    source_code: str,
    *,
    operation: str | None,
    message: str,
) -> IsolationDiagnosticCode:
    if source_code in _STABLE_CODE_VALUES:
        return IsolationDiagnosticCode(source_code)
    if source_code == "isolation.unsupported_syntax":
        return IsolationDiagnosticCode.UNSUPPORTED_MODE
    if source_code == "isolation.policy_widening":
        return IsolationDiagnosticCode.POLICY_DRIFT
    if source_code.startswith("isolation.review."):
        return IsolationDiagnosticCode.ELEVATION_REQUIRED
    if source_code.startswith("isolation.deny."):
        return IsolationDiagnosticCode.ELEVATION_DENIED
    if "approval_stale" in source_code or "approval.stale" in source_code:
        return IsolationDiagnosticCode.APPROVAL_STALE
    if source_code == "sandbox.backend.unavailable":
        return IsolationDiagnosticCode.SANDBOX_PROVIDER_UNAVAILABLE
    if source_code == "sandbox.backend.path_denied":
        return IsolationDiagnosticCode.SANDBOX_PATH_DENIED
    if source_code == "sandbox.backend.executable_denied":
        return IsolationDiagnosticCode.SANDBOX_PATH_DENIED
    if source_code == "sandbox.backend.output_rejected":
        return IsolationDiagnosticCode.SANDBOX_PATH_DENIED
    if source_code == "sandbox.backend.capability_mismatch":
        if "network" in message.lower():
            return IsolationDiagnosticCode.SANDBOX_NETWORK_UNENFORCEABLE
        return IsolationDiagnosticCode.CAPABILITY_MISMATCH
    if (
        source_code == "sandbox.backend.execution_failed"
        and operation == "prepare_profile"
    ):
        return IsolationDiagnosticCode.SANDBOX_PROFILE_GENERATION_FAILED
    if source_code in {
        "sandbox.backend.cancelled",
        "sandbox.backend.cleanup_failed",
        "sandbox.backend.concurrency_limit",
        "sandbox.backend.execution_failed",
        "sandbox.backend.stream_truncated",
        "sandbox.backend.timeout",
    }:
        return IsolationDiagnosticCode.MODE_UNAVAILABLE
    if source_code == "container.backend_required":
        return IsolationDiagnosticCode.UNSUPPORTED_BACKEND
    if source_code == "container.backend_unavailable":
        return IsolationDiagnosticCode.CONTAINER_BACKEND_UNAVAILABLE
    if source_code == "container.unsupported_syntax":
        return IsolationDiagnosticCode.UNSUPPORTED_MODE
    if source_code in {
        "container.backend.build_denied",
        "container.backend.capability_mismatch",
        "container.backend.image_denied",
        "container.backend.pool_denied",
        "container.backend.pull_denied",
        "container.backend.rootful_not_authorized",
    }:
        return IsolationDiagnosticCode.CONTAINER_BACKEND_CAPABILITY_MISMATCH
    if source_code.startswith("container.output."):
        return IsolationDiagnosticCode.CONTAINER_BACKEND_CAPABILITY_MISMATCH
    if source_code in {
        "container.backend.attach_failed",
        "container.backend.build_failed",
        "container.backend.cancelled",
        "container.backend.cleanup_failed",
        "container.backend.copy_failed",
        "container.backend.create_failed",
        "container.backend.event_dropped",
        "container.backend.orphan_quarantined",
        "container.backend.pull_failed",
        "container.backend.start_failed",
        "container.backend.stream_truncated",
        "container.backend.timeout",
        "container.backend.wait_failed",
    }:
        return IsolationDiagnosticCode.CONTAINER_BACKEND_UNAVAILABLE
    raise AssertionError(
        f"unsupported stable diagnostic source: {source_code}"
    )


def _combined_model_status(
    diagnostics: Sequence[IsolationMappedDiagnostic],
) -> str:
    statuses = {diagnostic.model_status for diagnostic in diagnostics}
    if "denied" in statuses:
        return "denied"
    if "unavailable" in statuses:
        return "unavailable"
    if "requires_review" in statuses:
        return "requires_review"
    return sorted(statuses)[0]


def _contains_marker(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)


def _contains_nonprintable(text: str) -> bool:
    return any(ord(char) < 32 and char not in "\n\r\t" for char in text)


def _redact_host_path_match(match: Match[str]) -> str:
    path = match.group("path").rstrip("/")
    name = path.rsplit("/", 1)[-1]
    return f"<host-path>/{name}" if name else "<host-path>"


_INVENTORY_ITEMS = (
    _inventory_item(
        IsolationDiagnosticCode.MODE_CONFLICT,
        category=IsolationDiagnosticCategory.VALUE,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Isolation mode settings conflict.",
        hint="Select one effective isolation mode.",
        audit_event="isolation.policy",
        model_status="denied",
    ),
    _inventory_item(
        IsolationDiagnosticCode.UNSUPPORTED_MODE,
        category=IsolationDiagnosticCategory.UNSUPPORTED,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Isolation mode is unsupported.",
        hint="Use a supported isolation mode.",
        audit_event="isolation.policy",
        model_status="denied",
    ),
    _inventory_item(
        IsolationDiagnosticCode.UNSUPPORTED_BACKEND,
        category=IsolationDiagnosticCategory.UNSUPPORTED,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Isolation backend is unsupported.",
        hint="Use a supported backend for the selected mode.",
        audit_event="isolation.backend",
        model_status="denied",
    ),
    _inventory_item(
        IsolationDiagnosticCode.MODE_UNAVAILABLE,
        category=IsolationDiagnosticCategory.AVAILABILITY,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Isolation mode is unavailable.",
        hint="Choose an available mode or install the required runtime.",
        audit_event="isolation.backend",
        model_status="unavailable",
    ),
    _inventory_item(
        IsolationDiagnosticCode.CAPABILITY_MISMATCH,
        category=IsolationDiagnosticCategory.SECURITY,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Isolation capabilities do not satisfy the policy.",
        hint="Use a backend that can enforce the requested controls.",
        audit_event="isolation.backend",
        model_status="denied",
    ),
    _inventory_item(
        IsolationDiagnosticCode.ELEVATION_REQUIRED,
        category=IsolationDiagnosticCategory.APPROVAL,
        severity=IsolationDiagnosticSeverity.WARNING,
        message="Isolation elevation requires approval.",
        hint="Request trusted approval before using this elevation.",
        audit_event="isolation.approval",
        model_status="requires_review",
    ),
    _inventory_item(
        IsolationDiagnosticCode.ELEVATION_DENIED,
        category=IsolationDiagnosticCategory.APPROVAL,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Isolation elevation was denied.",
        hint="Use a policy-approved isolation plan.",
        audit_event="isolation.approval",
        model_status="denied",
    ),
    _inventory_item(
        IsolationDiagnosticCode.FALLBACK_DENIED,
        category=IsolationDiagnosticCategory.SECURITY,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Isolation fallback was denied.",
        hint="Keep the required isolation mode or request approval.",
        audit_event="isolation.policy",
        model_status="denied",
    ),
    _inventory_item(
        IsolationDiagnosticCode.APPROVAL_STALE,
        category=IsolationDiagnosticCategory.APPROVAL,
        severity=IsolationDiagnosticSeverity.WARNING,
        message="Isolation approval is stale.",
        hint="Request a fresh approval for the current plan.",
        audit_event="isolation.approval",
        model_status="requires_review",
    ),
    _inventory_item(
        IsolationDiagnosticCode.POLICY_DRIFT,
        category=IsolationDiagnosticCategory.SECURITY,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Isolation policy drift was detected.",
        hint="Recompute the plan under the current policy.",
        audit_event="isolation.policy",
        model_status="denied",
    ),
    _inventory_item(
        IsolationDiagnosticCode.AUDIT_UNAVAILABLE,
        category=IsolationDiagnosticCategory.AUDIT,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Isolation audit recording is unavailable.",
        hint="Restore audit recording before executing this plan.",
        audit_event="isolation.audit",
        model_status="denied",
    ),
    _inventory_item(
        IsolationDiagnosticCode.SANDBOX_PROVIDER_UNAVAILABLE,
        category=IsolationDiagnosticCategory.AVAILABILITY,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Sandbox provider is unavailable.",
        hint="Install or enable the configured sandbox provider.",
        audit_event="sandbox.backend",
        model_status="unavailable",
    ),
    _inventory_item(
        IsolationDiagnosticCode.SANDBOX_PROFILE_GENERATION_FAILED,
        category=IsolationDiagnosticCategory.SECURITY,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Sandbox profile generation failed.",
        hint="Use a supported sandbox profile policy.",
        audit_event="sandbox.profile",
        model_status="denied",
    ),
    _inventory_item(
        IsolationDiagnosticCode.SANDBOX_PATH_DENIED,
        category=IsolationDiagnosticCategory.SECURITY,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Sandbox path access was denied.",
        hint="Use paths allowed by the sandbox policy.",
        audit_event="sandbox.filesystem",
        model_status="denied",
    ),
    _inventory_item(
        IsolationDiagnosticCode.SANDBOX_NETWORK_UNENFORCEABLE,
        category=IsolationDiagnosticCategory.SECURITY,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Sandbox network policy cannot be enforced.",
        hint="Use a backend that supports the requested network controls.",
        audit_event="sandbox.network",
        model_status="denied",
    ),
    _inventory_item(
        IsolationDiagnosticCode.CONTAINER_BACKEND_UNAVAILABLE,
        category=IsolationDiagnosticCategory.AVAILABILITY,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Container backend is unavailable.",
        hint="Install, start, or select an available container backend.",
        audit_event="container.backend",
        model_status="unavailable",
    ),
    _inventory_item(
        IsolationDiagnosticCode.CONTAINER_BACKEND_CAPABILITY_MISMATCH,
        category=IsolationDiagnosticCategory.SECURITY,
        severity=IsolationDiagnosticSeverity.ERROR,
        message="Container backend capabilities do not satisfy the policy.",
        hint="Select a backend that can enforce the requested controls.",
        audit_event="container.backend",
        model_status="denied",
    ),
)
_INVENTORY_BY_CODE = MappingProxyType(
    {
        cast(IsolationDiagnosticCode, item.code): item
        for item in _INVENTORY_ITEMS
    }
)
