"""Encode structured messages into a validated permanent-memory envelope."""

from ...entities import (
    Message,
    MessageContent,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageFile,
    MessageRole,
    MessageToolCall,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallDiagnosticStatus,
    ToolCallError,
    ToolCallResult,
    ToolValue,
    normalize_tool_arguments,
)

from datetime import datetime
from json import JSONDecodeError, dumps, loads
from math import isfinite
from typing import cast
from uuid import UUID

_ENVELOPE_PREFIX = "avalan-message-v1:"
_ENVELOPE_VERSION = 1


def encode_message_data(message: Message) -> str:
    """Encode one message without changing ordinary plain-text storage."""
    if not isinstance(message, Message):
        raise TypeError("message must be a message")
    if _is_plain_text_message(message) and not str(message.content).startswith(
        _ENVELOPE_PREFIX
    ):
        return str(message.content)
    payload: dict[str, object] = {
        "version": _ENVELOPE_VERSION,
        "role": message.role.value,
        "thinking": message.thinking,
        "content": _encode_content(message.content),
        "name": message.name,
        "arguments": (
            normalize_tool_arguments(message.arguments)
            if message.arguments is not None
            else None
        ),
        "tool_calls": (
            [_encode_message_tool_call(call) for call in message.tool_calls]
            if message.tool_calls is not None
            else None
        ),
        "tool_call_result": (
            _encode_tool_call_result(message.tool_call_result)
            if message.tool_call_result is not None
            else None
        ),
        "tool_call_error": (
            _encode_tool_call_error(message.tool_call_error)
            if message.tool_call_error is not None
            else None
        ),
        "tool_call_diagnostic": (
            _encode_tool_call_diagnostic(message.tool_call_diagnostic)
            if message.tool_call_diagnostic is not None
            else None
        ),
    }
    return _ENVELOPE_PREFIX + dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def decode_message_data(role: MessageRole, data: str) -> Message:
    """Decode one validated envelope or preserve legacy plain text."""
    if not isinstance(role, MessageRole):
        raise TypeError("role must be a message role")
    if not isinstance(data, str):
        raise TypeError("data must be a string")
    if not data.startswith(_ENVELOPE_PREFIX):
        return Message(role=role, content=data)
    encoded = data[len(_ENVELOPE_PREFIX) :]
    try:
        value: object = loads(encoded)
    except JSONDecodeError as exc:
        raise ValueError(
            "structured message envelope is invalid JSON"
        ) from exc
    payload = _object(value, "message")
    _require_keys(
        payload,
        {
            "version",
            "role",
            "thinking",
            "content",
            "name",
            "arguments",
            "tool_calls",
            "tool_call_result",
            "tool_call_error",
            "tool_call_diagnostic",
        },
        "message",
    )
    if payload["version"] != _ENVELOPE_VERSION:
        raise ValueError("structured message envelope version is unsupported")
    encoded_role = _string(payload["role"], "message.role")
    if encoded_role != role.value:
        raise ValueError("structured message role does not match its record")
    thinking = _optional_string(payload["thinking"], "message.thinking")
    name = _optional_string(payload["name"], "message.name")
    arguments = _optional_arguments(payload["arguments"], "message.arguments")
    tool_calls = _decode_message_tool_calls(payload["tool_calls"])
    return Message(
        role=role,
        thinking=thinking,
        content=_decode_content(payload["content"]),
        name=name,
        arguments=arguments,
        tool_calls=tool_calls,
        tool_call_result=_optional_tool_call_result(
            payload["tool_call_result"]
        ),
        tool_call_error=_optional_tool_call_error(payload["tool_call_error"]),
        tool_call_diagnostic=_optional_tool_call_diagnostic(
            payload["tool_call_diagnostic"]
        ),
    )


def message_partition_text(message: Message) -> str | None:
    """Return stable searchable text for one memory-bearing message."""
    if not _message_has_payload(message):
        return None
    return encode_message_data(message)


def _is_plain_text_message(message: Message) -> bool:
    return (
        isinstance(message.content, str)
        and message.thinking == ""
        and message.name is None
        and message.arguments is None
        and message.tool_calls is None
        and message.tool_call_result is None
        and message.tool_call_error is None
        and message.tool_call_diagnostic is None
    )


def _message_has_payload(message: Message) -> bool:
    return bool(
        message.content
        or message.thinking
        or message.name
        or message.arguments
        or message.tool_calls
        or message.tool_call_result
        or message.tool_call_error
        or message.tool_call_diagnostic
    )


def _encode_content(
    content: str | MessageContent | list[MessageContent] | None,
) -> object:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        return {
            "kind": "list",
            "items": [_encode_content_item(item) for item in content],
        }
    return _encode_content_item(content)


def _encode_content_item(content: MessageContent) -> dict[str, object]:
    if isinstance(content, MessageContentText):
        return {"kind": "text", "text": content.text}
    if isinstance(content, MessageContentImage):
        return {"kind": "image_url", "image_url": dict(content.image_url)}
    if isinstance(content, MessageContentFile):
        return {"kind": "file", "file": dict(content.file)}
    raise TypeError("message content contains an unsupported value")


def _decode_content(
    value: object,
) -> str | MessageContent | list[MessageContent] | None:
    if value is None or isinstance(value, str):
        return value
    payload = _object(value, "message.content")
    kind = _string(payload.get("kind"), "message.content.kind")
    if kind == "list":
        _require_keys(payload, {"kind", "items"}, "message.content")
        items = _list(payload["items"], "message.content.items")
        return [
            _decode_content_item(item, f"message.content.items[{index}]")
            for index, item in enumerate(items)
        ]
    return _decode_content_item(payload, "message.content")


def _decode_content_item(value: object, path: str) -> MessageContent:
    payload = _object(value, path)
    kind = _string(payload.get("kind"), f"{path}.kind")
    if kind == "text":
        _require_keys(payload, {"kind", "text"}, path)
        return MessageContentText(
            type="text",
            text=_string(payload["text"], f"{path}.text"),
        )
    if kind == "image_url":
        _require_keys(payload, {"kind", "image_url"}, path)
        image = _string_dict(payload["image_url"], f"{path}.image_url")
        return MessageContentImage(type="image_url", image_url=image)
    if kind == "file":
        _require_keys(payload, {"kind", "file"}, path)
        file_value = _object(payload["file"], f"{path}.file")
        text_fields = {
            "context",
            "data",
            "file_data",
            "file_id",
            "file_url",
            "filename",
            "local_path",
            "mime_type",
            "title",
            "url",
        }
        if not set(file_value) <= text_fields | {"citations"}:
            raise ValueError(f"{path}.file contains an unsupported field")
        file_data: dict[str, str | bool] = {}
        for key, item in file_value.items():
            if key == "citations":
                file_data[key] = _boolean(item, f"{path}.file.{key}")
            else:
                file_data[key] = _string(item, f"{path}.file.{key}")
        return MessageContentFile(
            type="file",
            file=cast(MessageFile, file_data),
        )
    raise ValueError(f"{path}.kind is unsupported")


def _encode_message_tool_call(call: MessageToolCall) -> dict[str, object]:
    return {
        "id": call.id,
        "name": call.name,
        "arguments": normalize_tool_arguments(call.arguments),
        "content_type": call.content_type,
    }


def _decode_message_tool_calls(value: object) -> list[MessageToolCall] | None:
    if value is None:
        return None
    items = _list(value, "message.tool_calls")
    calls: list[MessageToolCall] = []
    for index, item in enumerate(items):
        path = f"message.tool_calls[{index}]"
        payload = _object(item, path)
        _require_keys(
            payload,
            {"id", "name", "arguments", "content_type"},
            path,
        )
        content_type = _string(payload["content_type"], f"{path}.content_type")
        if content_type != "json":
            raise ValueError(f"{path}.content_type is unsupported")
        calls.append(
            MessageToolCall(
                id=_optional_string(payload["id"], f"{path}.id"),
                name=_string(payload["name"], f"{path}.name"),
                arguments=_arguments(
                    payload["arguments"], f"{path}.arguments"
                ),
            )
        )
    return calls


def _encode_identifier(value: UUID | str | None) -> object:
    if value is None:
        return None
    return {
        "kind": "uuid" if isinstance(value, UUID) else "str",
        "value": str(value),
    }


def _decode_identifier(value: object, path: str) -> UUID | str | None:
    if value is None:
        return None
    payload = _object(value, path)
    _require_keys(payload, {"kind", "value"}, path)
    kind = _string(payload["kind"], f"{path}.kind")
    item = _string(payload["value"], f"{path}.value")
    if kind == "str":
        return item
    if kind == "uuid":
        try:
            return UUID(item)
        except ValueError as exc:
            raise ValueError(f"{path}.value must be a UUID") from exc
    raise ValueError(f"{path}.kind is unsupported")


def _encode_tool_call(call: ToolCall) -> dict[str, object]:
    return {
        "id": _encode_identifier(call.id),
        "name": call.name,
        "arguments": (
            normalize_tool_arguments(call.arguments)
            if call.arguments is not None
            else None
        ),
        "provider_name": call.provider_name,
        "provider_name_encoded": call.provider_name_encoded,
        "provider_arguments_malformed": call.provider_arguments_malformed,
    }


def _decode_tool_call(value: object, path: str) -> ToolCall:
    payload = _object(value, path)
    _require_keys(
        payload,
        {
            "id",
            "name",
            "arguments",
            "provider_name",
            "provider_name_encoded",
            "provider_arguments_malformed",
        },
        path,
    )
    return ToolCall(
        id=_decode_identifier(payload["id"], f"{path}.id"),
        name=_string(payload["name"], f"{path}.name"),
        arguments=_optional_arguments(
            payload["arguments"], f"{path}.arguments"
        ),
        provider_name=_optional_string(
            payload["provider_name"],
            f"{path}.provider_name",
        ),
        provider_name_encoded=_boolean(
            payload["provider_name_encoded"],
            f"{path}.provider_name_encoded",
        ),
        provider_arguments_malformed=_boolean(
            payload["provider_arguments_malformed"],
            f"{path}.provider_arguments_malformed",
        ),
    )


def _encode_tool_call_result(result: ToolCallResult) -> dict[str, object]:
    return {
        "base": _encode_tool_call(result),
        "call": _encode_tool_call(result.call),
        "result": _json_value(result.result, "tool_call_result.result"),
    }


def _optional_tool_call_result(value: object) -> ToolCallResult | None:
    if value is None:
        return None
    payload = _object(value, "message.tool_call_result")
    _require_keys(
        payload, {"base", "call", "result"}, "message.tool_call_result"
    )
    base = _decode_tool_call(payload["base"], "message.tool_call_result.base")
    identifier = base.id
    if identifier is None:
        raise ValueError("message.tool_call_result.base.id is required")
    return ToolCallResult(
        id=identifier,
        name=base.name,
        arguments=base.arguments,
        provider_name=base.provider_name,
        provider_name_encoded=base.provider_name_encoded,
        provider_arguments_malformed=base.provider_arguments_malformed,
        call=_decode_tool_call(
            payload["call"], "message.tool_call_result.call"
        ),
        result=_json_value(
            payload["result"], "message.tool_call_result.result"
        ),
    )


def _encode_tool_call_error(error: ToolCallError) -> dict[str, object]:
    encoded_error: object = (
        {
            "kind": "exception",
            "type": error.error.__class__.__name__,
            "message": str(error.error),
        }
        if isinstance(error.error, BaseException)
        else {
            "kind": "json",
            "value": _json_value(error.error, "tool_call_error.error"),
        }
    )
    return {
        "base": _encode_tool_call(error),
        "call": _encode_tool_call(error.call),
        "error": encoded_error,
        "message": error.message,
    }


def _optional_tool_call_error(value: object) -> ToolCallError | None:
    if value is None:
        return None
    path = "message.tool_call_error"
    payload = _object(value, path)
    _require_keys(payload, {"base", "call", "error", "message"}, path)
    base = _decode_tool_call(payload["base"], f"{path}.base")
    identifier = base.id
    if identifier is None:
        raise ValueError(f"{path}.base.id is required")
    encoded_error = _object(payload["error"], f"{path}.error")
    kind = _string(encoded_error.get("kind"), f"{path}.error.kind")
    if kind == "json":
        _require_keys(encoded_error, {"kind", "value"}, f"{path}.error")
        error_value: ToolValue | BaseException = _json_value(
            encoded_error["value"],
            f"{path}.error.value",
        )
    elif kind == "exception":
        _require_keys(
            encoded_error,
            {"kind", "type", "message"},
            f"{path}.error",
        )
        error_value = {
            "type": _string(encoded_error["type"], f"{path}.error.type"),
            "message": _string(
                encoded_error["message"],
                f"{path}.error.message",
            ),
        }
    else:
        raise ValueError(f"{path}.error.kind is unsupported")
    return ToolCallError(
        id=identifier,
        name=base.name,
        arguments=base.arguments,
        provider_name=base.provider_name,
        provider_name_encoded=base.provider_name_encoded,
        provider_arguments_malformed=base.provider_arguments_malformed,
        call=_decode_tool_call(payload["call"], f"{path}.call"),
        error=error_value,
        message=_string(payload["message"], f"{path}.message"),
    )


def _encode_tool_call_diagnostic(
    diagnostic: ToolCallDiagnostic,
) -> dict[str, object]:
    return {
        "id": _encode_identifier(diagnostic.id),
        "call_id": _encode_identifier(diagnostic.call_id),
        "requested_name": diagnostic.requested_name,
        "canonical_name": diagnostic.canonical_name,
        "status": diagnostic.status.value,
        "code": diagnostic.code.value,
        "stage": diagnostic.stage.value,
        "message": diagnostic.message,
        "retryable": diagnostic.retryable,
        "details": normalize_tool_arguments(diagnostic.details),
        "started_at": (
            diagnostic.started_at.isoformat()
            if diagnostic.started_at is not None
            else None
        ),
        "finished_at": (
            diagnostic.finished_at.isoformat()
            if diagnostic.finished_at is not None
            else None
        ),
        "duration_ms": diagnostic.duration_ms,
    }


def _optional_tool_call_diagnostic(
    value: object,
) -> ToolCallDiagnostic | None:
    if value is None:
        return None
    path = "message.tool_call_diagnostic"
    payload = _object(value, path)
    _require_keys(
        payload,
        {
            "id",
            "call_id",
            "requested_name",
            "canonical_name",
            "status",
            "code",
            "stage",
            "message",
            "retryable",
            "details",
            "started_at",
            "finished_at",
            "duration_ms",
        },
        path,
    )
    identifier = _decode_identifier(payload["id"], f"{path}.id")
    if identifier is None:
        raise ValueError(f"{path}.id is required")
    return ToolCallDiagnostic(
        id=identifier,
        call_id=_decode_identifier(payload["call_id"], f"{path}.call_id"),
        requested_name=_optional_string(
            payload["requested_name"],
            f"{path}.requested_name",
        ),
        canonical_name=_optional_string(
            payload["canonical_name"],
            f"{path}.canonical_name",
        ),
        status=ToolCallDiagnosticStatus(
            _string(payload["status"], f"{path}.status")
        ),
        code=ToolCallDiagnosticCode(_string(payload["code"], f"{path}.code")),
        stage=ToolCallDiagnosticStage(
            _string(payload["stage"], f"{path}.stage")
        ),
        message=_string(payload["message"], f"{path}.message"),
        retryable=_boolean(payload["retryable"], f"{path}.retryable"),
        details=_arguments(payload["details"], f"{path}.details"),
        started_at=_optional_datetime(
            payload["started_at"], f"{path}.started_at"
        ),
        finished_at=_optional_datetime(
            payload["finished_at"],
            f"{path}.finished_at",
        ),
        duration_ms=_optional_number(
            payload["duration_ms"], f"{path}.duration_ms"
        ),
    )


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{path} must be an object with string keys")
    return {str(key): item for key, item in value.items()}


def _list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return list(value)


def _require_keys(
    value: dict[str, object],
    expected: set[str],
    path: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{path} fields do not match the envelope schema")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be text")
    return value


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be boolean")
    return value


def _string_dict(value: object, path: str) -> dict[str, str]:
    payload = _object(value, path)
    result: dict[str, str] = {}
    for key, item in payload.items():
        result[key] = _string(item, f"{path}.{key}")
    return result


def _arguments(value: object, path: str) -> dict[str, ToolValue]:
    try:
        return normalize_tool_arguments(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must contain JSON object values") from exc


def _optional_arguments(
    value: object,
    path: str,
) -> dict[str, ToolValue] | None:
    if value is None:
        return None
    return _arguments(value, path)


def _json_value(value: object, path: str) -> ToolValue:
    try:
        return normalize_tool_arguments({"value": value})["value"]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a JSON value") from exc


def _optional_datetime(value: object, path: str) -> datetime | None:
    if value is None:
        return None
    text = _string(value, path)
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{path} must be an ISO datetime") from exc


def _optional_number(value: object, path: str) -> int | float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{path} must be a number")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{path} must be finite")
    return value
