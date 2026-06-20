from ..entities import (
    ToolCall,
    ToolCallDiagnostic,
    ToolCallError,
    ToolCallOutcome,
    ToolCallResult,
    ToolValue,
)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from re import IGNORECASE, sub
from re import compile as compile_pattern
from typing import TypeAlias, cast, final

TOOL_DISPLAY_PROJECTION_METADATA_KEY = "tool_display_projection"
REDACTED_DISPLAY_VALUE = "[redacted]"
TRUNCATED_DISPLAY_SUFFIX = "..."
MAX_DISPLAY_LABEL_LENGTH = 80
MAX_DISPLAY_TEXT_LENGTH = 240
MAX_DISPLAY_DETAIL_VALUE_LENGTH = 240
MAX_DISPLAY_PREVIEW_LENGTH = 1000
MAX_DISPLAY_DETAILS = 12
MAX_DISPLAY_METRICS = 12
MAX_DISPLAY_SCAN_LENGTH = 1000

ToolDisplayScalar: TypeAlias = None | bool | int | float | str

_SENSITIVE_DISPLAY_PATTERN = compile_pattern(
    r"api[_-]?key|authorization|bearer|credential|password|passwd|"
    r"private[_-]?key|secret|session[_-]?key|token",
    IGNORECASE,
)
_ANSI_CONTROL_STRING_PATTERN = (
    r"(?:\x1b\][\s\S]*?(?:\x07|\x1b\\|$)"
    r"|\x9d[\s\S]*?(?:\x07|\x1b\\|\x9c|$)"
    r"|\x1b[PX^_][\s\S]*?(?:\x1b\\|$)"
    r"|[\x90\x98\x9e\x9f][\s\S]*?(?:\x1b\\|\x9c|$))"
)
_ANSI_CSI_PATTERN = r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]"
_ANSI_ESCAPE_PATTERN = r"\x1b[ -/]*[0-~]"


@dataclass(frozen=True, kw_only=True, slots=True)
class _SanitizedText:
    value: str
    redacted: bool = False
    truncated: bool = False


@dataclass(frozen=True, kw_only=True, slots=True)
class _SanitizedScalar:
    value: ToolDisplayScalar
    redacted: bool = False
    truncated: bool = False


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ToolDisplayDetail:
    label: str
    value: ToolDisplayScalar
    redacted: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        label = _sanitize_display_text(
            self.label,
            max_length=MAX_DISPLAY_LABEL_LENGTH,
        )
        value = _sanitize_display_scalar(
            label=self.label,
            value=self.value,
            max_length=MAX_DISPLAY_DETAIL_VALUE_LENGTH,
        )
        if label.redacted:
            value = _SanitizedScalar(
                value=REDACTED_DISPLAY_VALUE,
                redacted=True,
            )
        object.__setattr__(self, "label", label.value)
        object.__setattr__(self, "value", value.value)
        object.__setattr__(
            self,
            "redacted",
            self.redacted or label.redacted or value.redacted,
        )
        object.__setattr__(
            self,
            "truncated",
            self.truncated or label.truncated or value.truncated,
        )

    @classmethod
    def from_payload(cls, payload: object) -> "ToolDisplayDetail | None":
        if not isinstance(payload, Mapping):
            return None
        label = payload.get("label")
        if not isinstance(label, str):
            return None
        if "value" not in payload:
            return None
        value = payload["value"]
        if not _is_display_scalar(value):
            return None
        redacted = payload.get("redacted", False)
        truncated = payload.get("truncated", False)
        if not isinstance(redacted, bool) or not isinstance(truncated, bool):
            return None
        return cls(
            label=label,
            value=cast(ToolDisplayScalar, value),
            redacted=redacted,
            truncated=truncated,
        )

    def to_payload(self) -> dict[str, ToolValue]:
        payload: dict[str, ToolValue] = {
            "label": self.label,
            "value": self.value,
        }
        if self.redacted:
            payload["redacted"] = True
        if self.truncated:
            payload["truncated"] = True
        return payload


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ToolDisplayPreview:
    content: str
    label: str | None = None
    media_type: str | None = None
    redacted: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        content = _sanitize_display_text(
            self.content,
            max_length=MAX_DISPLAY_PREVIEW_LENGTH,
        )
        object.__setattr__(self, "content", content.value)
        label = _sanitize_optional_display_text(
            self.label,
            max_length=MAX_DISPLAY_LABEL_LENGTH,
        )
        if label is not None and label.redacted:
            content = _SanitizedText(
                value=REDACTED_DISPLAY_VALUE,
                redacted=True,
            )
        object.__setattr__(self, "label", label.value if label else None)
        object.__setattr__(self, "content", content.value)
        media_type = _sanitize_optional_display_text(
            self.media_type,
            max_length=MAX_DISPLAY_LABEL_LENGTH,
            redact_sensitive=False,
        )
        object.__setattr__(
            self,
            "media_type",
            media_type.value if media_type else None,
        )
        object.__setattr__(
            self,
            "redacted",
            self.redacted
            or content.redacted
            or (label.redacted if label else False),
        )
        object.__setattr__(
            self,
            "truncated",
            self.truncated
            or content.truncated
            or (label.truncated if label else False)
            or (media_type.truncated if media_type else False),
        )

    @classmethod
    def from_payload(cls, payload: object) -> "ToolDisplayPreview | None":
        if not isinstance(payload, Mapping):
            return None
        content = payload.get("content")
        if not isinstance(content, str):
            return None
        label = payload.get("label")
        media_type = payload.get("media_type")
        redacted = payload.get("redacted", False)
        truncated = payload.get("truncated", False)
        if label is not None and not isinstance(label, str):
            return None
        if media_type is not None and not isinstance(media_type, str):
            return None
        if not isinstance(redacted, bool) or not isinstance(truncated, bool):
            return None
        return cls(
            content=content,
            label=label,
            media_type=media_type,
            redacted=redacted,
            truncated=truncated,
        )

    def to_payload(self) -> dict[str, ToolValue]:
        payload: dict[str, ToolValue] = {"content": self.content}
        if self.label is not None:
            payload["label"] = self.label
        if self.media_type is not None:
            payload["media_type"] = self.media_type
        if self.redacted:
            payload["redacted"] = True
        if self.truncated:
            payload["truncated"] = True
        return payload


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ToolDisplayProjection:
    action: str
    label: str | None = None
    target: str | None = None
    scope: str | None = None
    summary: str | None = None
    status: str | None = None
    outcome: str | None = None
    severity: str | None = None
    progress: int | float | None = None
    details: Sequence[ToolDisplayDetail] = field(default_factory=tuple)
    metrics: Mapping[str, ToolDisplayScalar] = field(default_factory=dict)
    preview: ToolDisplayPreview | None = None
    redacted: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        action = _sanitize_display_text(
            self.action,
            max_length=MAX_DISPLAY_LABEL_LENGTH,
        )
        assert action.value, "action must not be empty"
        object.__setattr__(self, "action", action.value)

        redacted = self.redacted or action.redacted
        truncated = self.truncated or action.truncated
        for field_name in (
            "label",
            "target",
            "scope",
            "summary",
            "status",
            "outcome",
            "severity",
        ):
            text = _sanitize_optional_display_text(
                cast(str | None, getattr(self, field_name)),
                max_length=(
                    MAX_DISPLAY_LABEL_LENGTH
                    if field_name in {"label", "status", "outcome", "severity"}
                    else MAX_DISPLAY_TEXT_LENGTH
                ),
            )
            object.__setattr__(self, field_name, text.value if text else None)
            redacted = redacted or (text.redacted if text else False)
            truncated = truncated or (text.truncated if text else False)

        if self.progress is not None:
            assert isinstance(self.progress, int | float)
            assert not isinstance(self.progress, bool)
            assert isfinite(float(self.progress))
            assert 0 <= self.progress <= 1

        details, details_redacted, details_truncated = _sanitize_details(
            self.details,
        )
        object.__setattr__(self, "details", details)
        redacted = redacted or details_redacted
        truncated = truncated or details_truncated

        metrics, metrics_redacted, metrics_truncated = _sanitize_metrics(
            self.metrics,
        )
        object.__setattr__(self, "metrics", metrics)
        redacted = redacted or metrics_redacted
        truncated = truncated or metrics_truncated

        if self.preview is not None:
            assert isinstance(self.preview, ToolDisplayPreview)
            redacted = redacted or self.preview.redacted
            truncated = truncated or self.preview.truncated

        object.__setattr__(self, "redacted", redacted)
        object.__setattr__(self, "truncated", truncated)

    @classmethod
    def from_payload(cls, payload: object) -> "ToolDisplayProjection | None":
        if not isinstance(payload, Mapping):
            return None
        action = payload.get("action")
        if not isinstance(action, str):
            return None
        values = _projection_text_values(payload)
        if values is None:
            return None
        progress = payload.get("progress")
        if progress is not None and (
            not isinstance(progress, int | float)
            or isinstance(progress, bool)
            or not isfinite(float(progress))
        ):
            return None
        details_result = _details_from_payload(payload.get("details", []))
        if details_result is None:
            return None
        details, details_truncated = details_result
        metrics_result = _metrics_from_payload(payload.get("metrics", {}))
        if metrics_result is None:
            return None
        metrics, metrics_truncated = metrics_result
        preview = payload.get("preview")
        if preview is not None:
            if not isinstance(preview, Mapping):
                return None
            preview_value = ToolDisplayPreview.from_payload(preview)
            if preview_value is None:
                return None
        else:
            preview_value = None
        redacted = payload.get("redacted", False)
        truncated = payload.get("truncated", False)
        if not isinstance(redacted, bool) or not isinstance(truncated, bool):
            return None
        try:
            return cls(
                action=action,
                label=values["label"],
                target=values["target"],
                scope=values["scope"],
                summary=values["summary"],
                status=values["status"],
                outcome=values["outcome"],
                severity=values["severity"],
                progress=progress,
                details=details,
                metrics=metrics,
                preview=preview_value,
                redacted=redacted,
                truncated=truncated or details_truncated or metrics_truncated,
            )
        except AssertionError:
            return None

    def to_payload(self) -> dict[str, ToolValue]:
        payload: dict[str, ToolValue] = {"action": self.action}
        for key in (
            "label",
            "target",
            "scope",
            "summary",
            "status",
            "outcome",
            "severity",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = cast(ToolValue, value)
        if self.progress is not None:
            payload["progress"] = self.progress
        if self.details:
            payload["details"] = cast(
                ToolValue,
                [detail.to_payload() for detail in self.details],
            )
        if self.metrics:
            payload["metrics"] = cast(ToolValue, dict(self.metrics))
        if self.preview is not None:
            payload["preview"] = cast(ToolValue, self.preview.to_payload())
        if self.redacted:
            payload["redacted"] = True
        if self.truncated:
            payload["truncated"] = True
        assert is_json_safe_display_value(payload)
        return payload

    def to_metadata(self) -> dict[str, ToolValue]:
        return tool_display_projection_metadata(self)


def truncate_display_text(
    text: str,
    max_length: int = MAX_DISPLAY_TEXT_LENGTH,
) -> str:
    assert isinstance(text, str)
    assert isinstance(max_length, int)
    assert max_length > len(TRUNCATED_DISPLAY_SUFFIX)
    truncated = len(text) > max_length
    source = text[: max_length + len(TRUNCATED_DISPLAY_SUFFIX)]
    normalized = _normalize_display_text(source)
    if len(normalized) <= max_length:
        return (
            normalized
            if not truncated
            else normalized.rstrip() + TRUNCATED_DISPLAY_SUFFIX
        )[:max_length]
    return (
        normalized[: max_length - len(TRUNCATED_DISPLAY_SUFFIX)].rstrip()
        + TRUNCATED_DISPLAY_SUFFIX
    )


def is_sensitive_display_label(label: str) -> bool:
    assert isinstance(label, str)
    return _contains_sensitive_text(label)


def is_sensitive_display_value(value: object) -> bool:
    return _contains_sensitive_display_value(value)


def sanitize_display_label(label: str) -> str:
    assert isinstance(label, str)
    return _sanitize_display_text(
        label,
        max_length=MAX_DISPLAY_LABEL_LENGTH,
    ).value


def sanitize_display_value(
    label: str | None,
    value: object,
    max_length: int = MAX_DISPLAY_DETAIL_VALUE_LENGTH,
) -> ToolDisplayScalar:
    if label is not None:
        assert isinstance(label, str)
    return _sanitize_display_scalar(
        label=label,
        value=value,
        max_length=max_length,
    ).value


def is_json_safe_display_value(value: object) -> bool:
    if value is None or isinstance(value, bool | str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return isfinite(value)
    if isinstance(value, list):
        return all(is_json_safe_display_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and is_json_safe_display_value(item)
            for key, item in value.items()
        )
    return False


def tool_display_projection_metadata(
    projection: ToolDisplayProjection,
) -> dict[str, ToolValue]:
    assert isinstance(projection, ToolDisplayProjection)
    payload = projection.to_payload()
    return {
        TOOL_DISPLAY_PROJECTION_METADATA_KEY: cast(ToolValue, payload),
    }


def tool_display_projection_from_metadata(
    metadata: Mapping[str, object] | None,
) -> ToolDisplayProjection | None:
    if metadata is None:
        return None
    assert isinstance(metadata, Mapping)
    payload = metadata.get(TOOL_DISPLAY_PROJECTION_METADATA_KEY)
    if not isinstance(payload, Mapping):
        return None
    projection = ToolDisplayProjection.from_payload(payload)
    if projection is None:
        return None
    return projection


def tool_call_display_projection_from_metadata(
    call: ToolCall,
    metadata: Mapping[str, object] | None,
) -> ToolDisplayProjection:
    assert isinstance(call, ToolCall)
    projection = tool_display_projection_from_metadata(metadata)
    return projection or fallback_tool_call_display_projection(call)


def tool_outcome_display_projection_from_metadata(
    outcome: ToolCallOutcome,
    metadata: Mapping[str, object] | None,
) -> ToolDisplayProjection:
    projection = tool_display_projection_from_metadata(metadata)
    return projection or fallback_tool_outcome_display_projection(outcome)


def fallback_tool_call_display_projection(
    call: ToolCall,
) -> ToolDisplayProjection:
    assert isinstance(call, ToolCall)
    tool_name = call.name.strip() or "tool"
    return ToolDisplayProjection(
        action="call",
        label=tool_name,
        target=tool_name,
        summary=f"Call {tool_name}.",
        details=_details_from_arguments(call.arguments),
    )


def fallback_tool_outcome_display_projection(
    outcome: ToolCallOutcome,
) -> ToolDisplayProjection:
    if isinstance(outcome, ToolCallResult):
        tool_name = _outcome_tool_name(outcome.call)
        return ToolDisplayProjection(
            action="finish",
            label=tool_name,
            target=tool_name,
            summary=f"{tool_name} completed.",
            status="completed",
            outcome="result",
            details=_details_from_arguments(outcome.call.arguments),
            preview=_preview_from_value("Result", outcome.result),
        )
    if isinstance(outcome, ToolCallError):
        tool_name = _outcome_tool_name(outcome.call)
        return ToolDisplayProjection(
            action="finish",
            label=tool_name,
            target=tool_name,
            summary=outcome.message,
            status="error",
            outcome="error",
            severity="error",
            details=(
                ToolDisplayDetail(
                    label="error_type",
                    value=outcome.error_type,
                ),
            ),
            preview=_preview_from_value("Error", outcome.error),
        )
    assert isinstance(outcome, ToolCallDiagnostic)
    tool_name = outcome.canonical_name or outcome.requested_name or "tool"
    return ToolDisplayProjection(
        action="skip",
        label=tool_name,
        target=tool_name,
        summary=outcome.message,
        status=outcome.status.value,
        outcome=outcome.code.value,
        severity="warning",
        details=(
            ToolDisplayDetail(label="stage", value=outcome.stage.value),
            ToolDisplayDetail(label="retryable", value=outcome.retryable),
        ),
    )


def _normalize_display_text(text: str) -> str:
    assert isinstance(text, str)
    text = _strip_display_controls(text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    return " ".join(text.split())


def _strip_display_controls(text: str) -> str:
    assert isinstance(text, str)
    text = sub(_ANSI_CONTROL_STRING_PATTERN, "", text)
    text = sub(_ANSI_CSI_PATTERN, "", text)
    return sub(_ANSI_ESCAPE_PATTERN, "", text)


def _contains_sensitive_text(text: str) -> bool:
    assert isinstance(text, str)
    sampled, _ = _bounded_text_sample(text, MAX_DISPLAY_SCAN_LENGTH)
    key_text = _strip_display_controls(sampled).lower()
    key_text = sub(r"\[/?[^\]]+\]", "", key_text)
    compact_key = sub(r"[^a-z0-9]", "", key_text)
    return bool(_SENSITIVE_DISPLAY_PATTERN.search(key_text)) or any(
        sensitive.replace("_", "") in compact_key
        for sensitive in (
            "api_key",
            "authorization",
            "bearer",
            "credential",
            "password",
            "passwd",
            "private_key",
            "secret",
            "session_key",
            "token",
        )
    )


def _sanitize_display_text(
    text: str,
    *,
    max_length: int,
    redact_sensitive: bool = True,
) -> _SanitizedText:
    assert isinstance(text, str)
    assert isinstance(max_length, int)
    assert max_length > len(TRUNCATED_DISPLAY_SUFFIX)
    if redact_sensitive and _contains_sensitive_text(text):
        return _SanitizedText(value=REDACTED_DISPLAY_VALUE, redacted=True)
    source, source_truncated = _bounded_text_sample(text, max_length)
    normalized = _normalize_display_text(source)
    truncated = source_truncated or len(normalized) > max_length
    return _SanitizedText(
        value=truncate_display_text(normalized, max_length),
        truncated=truncated,
    )


def _sanitize_optional_display_text(
    text: str | None,
    *,
    max_length: int,
    redact_sensitive: bool = True,
) -> _SanitizedText | None:
    if text is None:
        return None
    assert isinstance(text, str)
    sanitized = _sanitize_display_text(
        text,
        max_length=max_length,
        redact_sensitive=redact_sensitive,
    )
    if not sanitized.value:
        return None
    return sanitized


def _sanitize_display_scalar(
    *,
    label: str | None,
    value: object,
    max_length: int,
) -> _SanitizedScalar:
    if label is not None:
        assert isinstance(label, str)
    assert isinstance(max_length, int)
    assert max_length > len(TRUNCATED_DISPLAY_SUFFIX)
    if (
        label is not None
        and is_sensitive_display_label(label)
        or is_sensitive_display_value(value)
    ):
        return _SanitizedScalar(
            value=REDACTED_DISPLAY_VALUE,
            redacted=True,
        )
    if value is None:
        return _SanitizedScalar(value=None)
    if isinstance(value, bool):
        return _SanitizedScalar(value=value)
    if isinstance(value, int):
        return _SanitizedScalar(value=value)
    if isinstance(value, float) and isfinite(value):
        return _SanitizedScalar(value=value)
    if isinstance(value, str):
        text = _sanitize_display_text(value, max_length=max_length)
        return _SanitizedScalar(
            value=text.value,
            redacted=text.redacted,
            truncated=text.truncated,
        )
    bounded = _bounded_display_text(value, max_length)
    text = _sanitize_display_text(
        bounded,
        max_length=max_length,
    )
    return _SanitizedScalar(
        value=text.value,
        redacted=text.redacted,
        truncated=text.truncated or bounded.endswith(TRUNCATED_DISPLAY_SUFFIX),
    )


def _contains_sensitive_display_value(value: object, depth: int = 0) -> bool:
    if depth > 4:
        return False
    if isinstance(value, str):
        return _contains_sensitive_text(value)
    if isinstance(value, Mapping):
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_DISPLAY_DETAILS:
                break
            if isinstance(key, str) and is_sensitive_display_label(key):
                return True
            if _contains_sensitive_display_value(item, depth + 1):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    ):
        for index, item in enumerate(value):
            if index >= MAX_DISPLAY_DETAILS:
                break
            if _contains_sensitive_display_value(item, depth + 1):
                return True
        return False
    return False


def _is_display_scalar(value: object) -> bool:
    if value is None or isinstance(value, bool | str):
        return True
    if isinstance(value, int):
        return True
    return isinstance(value, float) and isfinite(value)


def _sanitize_details(
    details: Sequence[ToolDisplayDetail],
) -> tuple[tuple[ToolDisplayDetail, ...], bool, bool]:
    assert isinstance(details, Sequence)
    assert not isinstance(details, str | bytes | bytearray)
    selected = tuple(details[:MAX_DISPLAY_DETAILS])
    redacted = False
    truncated = len(details) > MAX_DISPLAY_DETAILS
    for detail in selected:
        assert isinstance(detail, ToolDisplayDetail)
        redacted = redacted or detail.redacted
        truncated = truncated or detail.truncated
    return selected, redacted, truncated


def _sanitize_metrics(
    metrics: Mapping[str, ToolDisplayScalar],
) -> tuple[dict[str, ToolDisplayScalar], bool, bool]:
    assert isinstance(metrics, Mapping)
    result: dict[str, ToolDisplayScalar] = {}
    redacted = False
    truncated = len(metrics) > MAX_DISPLAY_METRICS
    for index, (label, value) in enumerate(metrics.items()):
        if index >= MAX_DISPLAY_METRICS:
            break
        assert isinstance(label, str)
        assert _is_display_scalar(value)
        sanitized_label = _sanitize_display_text(
            label,
            max_length=MAX_DISPLAY_LABEL_LENGTH,
        )
        sanitized_value = _sanitize_display_scalar(
            label=label,
            value=value,
            max_length=MAX_DISPLAY_DETAIL_VALUE_LENGTH,
        )
        if sanitized_label.redacted:
            sanitized_value = _SanitizedScalar(
                value=REDACTED_DISPLAY_VALUE,
                redacted=True,
            )
        result[sanitized_label.value] = sanitized_value.value
        redacted = (
            redacted or sanitized_label.redacted or sanitized_value.redacted
        )
        truncated = (
            truncated or sanitized_label.truncated or sanitized_value.truncated
        )
    return result, redacted, truncated


def _projection_text_values(
    payload: Mapping[str, object],
) -> dict[str, str | None] | None:
    values: dict[str, str | None] = {}
    for key in (
        "label",
        "target",
        "scope",
        "summary",
        "status",
        "outcome",
        "severity",
    ):
        value = payload.get(key)
        if value is not None and not isinstance(value, str):
            return None
        values[key] = value
    return values


def _details_from_payload(
    value: object,
) -> tuple[tuple[ToolDisplayDetail, ...], bool] | None:
    if not isinstance(value, Sequence) or isinstance(
        value, str | bytes | bytearray
    ):
        return None
    details: list[ToolDisplayDetail] = []
    truncated = False
    for index, item in enumerate(value):
        if index >= MAX_DISPLAY_DETAILS:
            truncated = True
            break
        if not isinstance(item, Mapping):
            return None
        detail = ToolDisplayDetail.from_payload(item)
        if detail is None:
            return None
        details.append(detail)
    return tuple(details), truncated


def _metrics_from_payload(
    value: object,
) -> tuple[dict[str, ToolDisplayScalar], bool] | None:
    if not isinstance(value, Mapping):
        return None
    metrics: dict[str, ToolDisplayScalar] = {}
    truncated = False
    for index, (key, item) in enumerate(value.items()):
        if index >= MAX_DISPLAY_METRICS:
            truncated = True
            break
        if not isinstance(key, str):
            return None
        if not _is_display_scalar(item):
            return None
        metrics[key] = item
    return metrics, truncated


def _details_from_arguments(
    arguments: Mapping[str, object] | None,
) -> tuple[ToolDisplayDetail, ...]:
    if not arguments:
        return ()
    details: list[ToolDisplayDetail] = []
    for index, (label, value) in enumerate(arguments.items()):
        if index >= MAX_DISPLAY_DETAILS:
            break
        details.append(
            ToolDisplayDetail(
                label=label,
                value=sanitize_display_value(label, value),
            )
        )
    return tuple(details)


def _preview_from_value(
    label: str,
    value: object,
) -> ToolDisplayPreview | None:
    if value is None:
        return None
    return ToolDisplayPreview(
        content=_bounded_display_text(value, MAX_DISPLAY_PREVIEW_LENGTH),
        label=label,
    )


def _outcome_tool_name(call: ToolCall) -> str:
    assert isinstance(call, ToolCall)
    return call.name.strip() or "tool"


def _bounded_display_text(
    value: object,
    max_length: int = MAX_DISPLAY_TEXT_LENGTH,
) -> str:
    assert isinstance(max_length, int)
    assert max_length > len(TRUNCATED_DISPLAY_SUFFIX)
    return _bounded_display_text_part(value, max_length, 0)


def _bounded_text_sample(text: str, max_length: int) -> tuple[str, bool]:
    assert isinstance(text, str)
    assert isinstance(max_length, int)
    assert max_length > len(TRUNCATED_DISPLAY_SUFFIX)
    sample_length = max(
        max_length * 2,
        max_length + len(TRUNCATED_DISPLAY_SUFFIX),
    )
    if len(text) <= sample_length:
        return text, False
    edge_length = sample_length // 2
    return f"{text[:edge_length]} {text[-edge_length:]}", True


def _bounded_display_text_part(
    value: object,
    remaining: int,
    depth: int,
) -> str:
    if remaining <= len(TRUNCATED_DISPLAY_SUFFIX):
        return TRUNCATED_DISPLAY_SUFFIX
    if value is None or isinstance(value, bool | int | float):
        return truncate_display_text(str(value), remaining)
    if isinstance(value, str):
        return truncate_display_text(value, remaining)
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, Mapping):
        return _bounded_mapping_text(value, remaining, depth)
    if isinstance(value, Sequence) and not isinstance(
        value, bytes | bytearray
    ):
        return _bounded_sequence_text(value, remaining, depth)
    return type(value).__name__


def _bounded_mapping_text(
    value: Mapping[object, object],
    remaining: int,
    depth: int,
) -> str:
    parts: list[str] = ["{"]
    for index, (key, item) in enumerate(value.items()):
        if index >= MAX_DISPLAY_DETAILS:
            _append_bounded_part(parts, ", ...", remaining)
            break
        separator = "" if index == 0 else ", "
        key_text = _bounded_display_text_part(key, 48, depth + 1)
        item_text = _bounded_display_text_part(item, 80, depth + 1)
        if not _append_bounded_part(
            parts,
            f"{separator}{key_text}: {item_text}",
            remaining,
        ):
            break
    _append_bounded_part(parts, "}", remaining)
    return truncate_display_text("".join(parts), remaining)


def _bounded_sequence_text(
    value: Sequence[object],
    remaining: int,
    depth: int,
) -> str:
    parts: list[str] = ["["]
    for index, item in enumerate(value[:MAX_DISPLAY_DETAILS]):
        separator = "" if index == 0 else ", "
        item_text = _bounded_display_text_part(item, 80, depth + 1)
        if not _append_bounded_part(
            parts,
            f"{separator}{item_text}",
            remaining,
        ):
            break
    if len(value) > MAX_DISPLAY_DETAILS:
        _append_bounded_part(parts, ", ...", remaining)
    _append_bounded_part(parts, "]", remaining)
    return truncate_display_text("".join(parts), remaining)


def _append_bounded_part(
    parts: list[str],
    text: str,
    remaining: int,
) -> bool:
    current_length = sum(len(part) for part in parts)
    if current_length + len(text) <= remaining:
        parts.append(text)
        return True
    available = remaining - current_length
    if available > len(TRUNCATED_DISPLAY_SUFFIX):
        parts.append(truncate_display_text(text, available))
    return False
