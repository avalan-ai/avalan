from ..entities import (
    MessageRole,
    OrchestratorSettings,
    ReasoningEffort,
    ToolNamePolicySettings,
)
from ..server_output_redaction import (
    SERVER_OUTPUT_REDACTION_CHANNELS as SERVER_OUTPUT_REDACTION_CHANNELS,
)
from ..server_output_redaction import (
    SERVER_OUTPUT_REDACTION_PROTOCOLS as SERVER_OUTPUT_REDACTION_PROTOCOLS,
)
from ..server_output_redaction import (
    SERVER_OUTPUT_REDACTION_RULES as SERVER_OUTPUT_REDACTION_RULES,
)
from ..server_output_redaction import (
    ServerOutputRedactionChannel as ServerOutputRedactionChannel,
)
from ..server_output_redaction import (
    ServerOutputRedactionProtocol as ServerOutputRedactionProtocol,
)
from ..server_output_redaction import (
    ServerOutputRedactionRule as ServerOutputRedactionRule,
)
from ..skill import (
    SKILL_MAIN_RESOURCE_FILENAME,
    SkillModelValue,
    TrustedSkillSettings,
    contains_skill_source_resource_reference,
    could_contain_skill_source_path_reference_prefix,
    trusted_skill_settings_fingerprint,
    trusted_skill_source_fingerprint,
    trusted_skill_source_identity_dict,
)
from ..tool.context import ToolSettingsContext
from .authority import (
    reject_remote_runtime_authority_extra_fields,
    reject_remote_runtime_authority_fields,
    reject_remote_runtime_authority_model_identifier,
)

from base64 import b64decode
from binascii import Error as BinasciiError
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from re import Match, fullmatch
from re import compile as compile_pattern
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

JSONType = Literal["bool", "float", "int", "object", "string"]
MCP_FILE_DATA_KEYS = ("data", "base64", "file_data")
MCP_FILE_URL_KEYS = ("uri", "url", "file_url")
MCP_FILE_SOURCE_KEYS = MCP_FILE_DATA_KEYS + MCP_FILE_URL_KEYS
MCP_FILE_MIME_TYPE_KEYS = ("mimeType", "mime_type")
MCP_FILE_FILENAME_KEYS = (
    "filename",
    "fileName",
    "file_name",
    "name",
    "displayName",
)
MCP_BASE64_SOURCE_PATTERN = (
    r"^(?:data:[^,]+,\s*)?"
    r"(?:(?:[A-Za-z0-9+/]{4})+"
    r"(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?"
    r"|[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)$"
)
NON_WHITESPACE_PATTERN = r".*\S.*"
SKILL_CONTENT_REDACTION = "<redacted-skill-content>"
HOST_PATH_REDACTION = "<host-path>"
_STREAM_HOST_PATH_POSIX_ROOTS = (
    "/Users",
    "/home",
    "/private",
    "/secret",
    "/var",
    "/etc",
    "/root",
    "/tmp",
    "/opt",
    "/Volumes",
)
_HOST_PATH_PATTERN = compile_pattern(
    r"(?P<path>"
    r"(?:/Users|/home|/private|/secret|/var|/etc|/root|/tmp|/opt|/Volumes)"
    r"(?:/+[^/\s,;:\"'\\\]}]+)*(?:/+)?"
    r"|[A-Za-z]:\\Users\\[^\s,;:\"'\\\]}]+"
    r"(?:\\[^\s,;:\"'\\\]}]+)*"
    r"|[A-Za-z]:/Users(?:/+[^/\s,;:\"'\\\]}]+)*(?:/+)?"
    r")"
    r"(?=$|[\s,;:\"'\\\]}])"
)
_SKILL_BODY_PHRASE_MARKERS = (
    "use when",
    "trigger rules",
    "how to use",
    "instructions:",
)
_SKILL_BODY_RESOURCE_FIELD_MARKERS = ("source:", "path:", "file:")
_SKILL_BODY_HOST_PATH_PREFIX_MARKERS = (
    "/users",
    "/home",
    "/private",
    "/tmp",
    "c:\\users",
    "c:/users",
)
_SKILL_BODY_TEXT_FOLLOWUP_MARKERS = (
    *_SKILL_BODY_PHRASE_MARKERS,
    *_SKILL_BODY_RESOURCE_FIELD_MARKERS,
    SKILL_MAIN_RESOURCE_FILENAME.lower(),
    *_SKILL_BODY_HOST_PATH_PREFIX_MARKERS,
)
_SKILL_BODY_DETECTION_WINDOW = 12000
_SKILL_BODY_HEADING_PATTERN = compile_pattern(
    r"(?is)(?:^|\n)\s*#\s+(?P<title>[^\n]{1,160})\n\s*\n"
    r".{0,12000}\b"
    r"(?:use when|trigger rules|how to use|instructions\s*:)"
)
_SKILL_BODY_RESOURCE_START_PATTERN = compile_pattern(
    r"(?is)(?:^|\n)\s*(?:#\s+[^\n]{1,160}|---\s*\n|description\s*:)"
)
_SKILL_BODY_STREAM_START_PATTERN = compile_pattern(
    r"(?is)(?:^|\n)\s*(?:#\s+|---\s*(?:\n|$)|description\s*:)"
)
_SKILL_BODY_HEADING_SEPARATOR_PATTERN = compile_pattern(r"\s*\n")
_URL_SCHEME_PATTERN = compile_pattern(r"(?i)[A-Za-z][A-Za-z0-9+.-]*:")
_SKILLS_TOOL_PREFIX = "skills."
_SKILL_CONTENT_KEYS = frozenset({"content"})
_ORDINARY_TITLE_HEADING_WORDS = frozenset(
    {
        "answer",
        "conclusion",
        "intro",
        "introduction",
        "notes",
        "overview",
        "plan",
        "reason",
        "reasoning",
        "report",
        "results",
        "summary",
        "update",
    }
)


@dataclass(kw_only=True, frozen=True)
class ServerOutputRedactionSettings:
    """Control optional server output redaction."""

    enabled: bool = False
    rules: frozenset[ServerOutputRedactionRule] = field(
        default_factory=lambda: frozenset(SERVER_OUTPUT_REDACTION_RULES)
    )
    protocols: frozenset[ServerOutputRedactionProtocol] = field(
        default_factory=lambda: frozenset(SERVER_OUTPUT_REDACTION_PROTOCOLS)
    )
    channels: frozenset[ServerOutputRedactionChannel] = field(
        default_factory=lambda: frozenset(SERVER_OUTPUT_REDACTION_CHANNELS)
    )

    def __post_init__(self) -> None:
        assert isinstance(self.enabled, bool)
        rules = frozenset(self.rules)
        protocols = frozenset(self.protocols)
        channels = frozenset(self.channels)
        assert rules <= frozenset(SERVER_OUTPUT_REDACTION_RULES)
        assert protocols <= frozenset(SERVER_OUTPUT_REDACTION_PROTOCOLS)
        assert channels <= frozenset(SERVER_OUTPUT_REDACTION_CHANNELS)
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "protocols", protocols)
        object.__setattr__(self, "channels", channels)

    def should_redact(
        self,
        rule: ServerOutputRedactionRule,
        *,
        protocol: ServerOutputRedactionProtocol | None = None,
        channel: ServerOutputRedactionChannel | None = None,
    ) -> bool:
        """Return whether a redaction rule applies."""
        assert rule in SERVER_OUTPUT_REDACTION_RULES
        if protocol is not None:
            assert protocol in SERVER_OUTPUT_REDACTION_PROTOCOLS
        if channel is not None:
            assert channel in SERVER_OUTPUT_REDACTION_CHANNELS
        channel_applies = (
            self.channels == frozenset(SERVER_OUTPUT_REDACTION_CHANNELS)
            if channel is None
            else channel in self.channels
        )
        return (
            self.enabled
            and rule in self.rules
            and (protocol is None or protocol in self.protocols)
            and channel_applies
        )


def coerce_server_output_redaction_settings(
    value: object | None,
) -> ServerOutputRedactionSettings:
    """Return a server output redaction settings instance."""
    return (
        value
        if isinstance(value, ServerOutputRedactionSettings)
        else ServerOutputRedactionSettings()
    )


def server_output_redaction_settings_from_state(
    state: object,
) -> ServerOutputRedactionSettings:
    """Return server output redaction settings from app state."""
    ctx = getattr(state, "ctx", None)
    ctx_settings = getattr(ctx, "output_redaction_settings", None)
    if isinstance(ctx_settings, ServerOutputRedactionSettings):
        return ctx_settings
    settings = getattr(state, "server_output_redaction_settings", None)
    return coerce_server_output_redaction_settings(settings)


def _has_non_empty_file_source(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if not value.strip():
        return False
    if value.startswith("data:"):
        _, separator, payload = value.rpartition(",")
        return bool(separator and payload.strip())
    return True


def _present_file_source_keys(value: dict[Any, Any]) -> list[str]:
    return [key for key in MCP_FILE_SOURCE_KEYS if key in value]


def _present_keys(value: dict[Any, Any], keys: tuple[str, ...]) -> list[str]:
    return [key for key in keys if key in value]


def _validate_base64_file_source(value: str) -> str:
    source = value.strip()
    payload = source
    if payload.startswith("data:"):
        prefix, separator, payload = payload.rpartition(",")
        if not separator:
            raise ValueError("File descriptor data URL must include payload")
        payload = payload.strip()
        source = f"{prefix},{payload}"
    if fullmatch(MCP_BASE64_SOURCE_PATTERN, source) is None:
        raise ValueError("File descriptor data must be base64")
    try:
        b64decode(payload, validate=True)
    except (BinasciiError, ValueError) as exc:
        raise ValueError("File descriptor data must be base64") from exc
    return source


def _schema_property(properties: dict[str, Any], key: str) -> dict[str, Any]:
    property_schema = properties.get(key)
    if isinstance(property_schema, dict):
        return dict(property_schema)
    return {"type": "string"}


def _non_empty_string_schema(
    properties: dict[str, Any], key: str
) -> dict[str, Any]:
    property_schema = _schema_property(properties, key)
    schema: dict[str, Any] = {
        "type": "string",
        "minLength": 1,
        "pattern": NON_WHITESPACE_PATTERN,
    }
    for metadata_key in ("description", "title"):
        if metadata_key in property_schema:
            schema[metadata_key] = property_schema[metadata_key]
    return schema


def _mcp_file_descriptor_json_schema(schema: dict[str, Any]) -> None:
    properties = schema.setdefault("properties", {})
    assert isinstance(properties, dict)

    data_schema = _non_empty_string_schema(properties, "data")
    data_schema["pattern"] = MCP_BASE64_SOURCE_PATTERN
    for key in MCP_FILE_DATA_KEYS:
        properties[key] = dict(data_schema)

    uri_schema = _non_empty_string_schema(properties, "uri")
    for key in MCP_FILE_URL_KEYS:
        properties[key] = dict(uri_schema)

    mime_type_schema = _schema_property(properties, "mimeType")
    properties["mimeType"] = _non_empty_string_schema(properties, "mimeType")
    properties["mime_type"] = _non_empty_string_schema(
        {"mime_type": mime_type_schema}, "mime_type"
    )
    filename_schema = _schema_property(properties, "filename")
    for key in MCP_FILE_FILENAME_KEYS:
        properties[key] = _non_empty_string_schema({key: filename_schema}, key)
    schema["anyOf"] = [{"required": [key]} for key in MCP_FILE_SOURCE_KEYS]
    schema["not"] = {
        "anyOf": [
            {"required": [left, right]}
            for index, left in enumerate(MCP_FILE_SOURCE_KEYS)
            for right in MCP_FILE_SOURCE_KEYS[index + 1 :]
        ]
    }


def _mcp_tool_request_json_schema(schema: dict[str, Any]) -> None:
    properties = schema.setdefault("properties", {})
    assert isinstance(properties, dict)
    files_schema = _schema_property(properties, "files")
    for key in ("input_files", "file_descriptors"):
        properties[key] = dict(files_schema)
    file_keys = ("files", "input_files", "file_descriptors")
    schema["anyOf"] = [
        {
            "required": ["input_string"],
            "properties": {
                "input_string": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": NON_WHITESPACE_PATTERN,
                }
            },
        },
        *[
            {
                "required": [key],
                "properties": {key: {"type": "array", "minItems": 1}},
            }
            for key in file_keys
        ],
    ]
    schema["not"] = {
        "anyOf": [
            {"required": [left, right]}
            for index, left in enumerate(file_keys)
            for right in file_keys[index + 1 :]
        ]
    }


@dataclass(kw_only=True, frozen=True)
class OrchestratorContext:
    participant_id: UUID
    specs_path: str | None = None
    settings: OrchestratorSettings | None = None
    tool_settings: ToolSettingsContext | None = None
    tool_name_policy: ToolNamePolicySettings | None = None
    output_redaction_settings: ServerOutputRedactionSettings = field(
        default_factory=ServerOutputRedactionSettings
    )
    skills_settings: TrustedSkillSettings | None = None
    skills_registry_metadata: Mapping[str, SkillModelValue] | None = None

    def __post_init__(self) -> None:
        assert isinstance(
            self.output_redaction_settings,
            ServerOutputRedactionSettings,
        )
        if self.skills_settings is not None:
            assert isinstance(self.skills_settings, TrustedSkillSettings)
        if self.skills_registry_metadata is not None:
            assert isinstance(self.skills_registry_metadata, Mapping)


def server_skills_registry_metadata(
    settings: TrustedSkillSettings | None,
) -> dict[str, SkillModelValue] | None:
    """Return logical skills metadata for server runtime boundaries."""
    if settings is None:
        return None
    assert isinstance(settings, TrustedSkillSettings)
    return {
        "enabled": settings.enabled,
        "settings_fingerprint": trusted_skill_settings_fingerprint(settings),
        "source_fingerprint": trusted_skill_source_fingerprint(settings),
        "source_labels": tuple(source.label for source in settings.sources),
        "authority_kinds": tuple(
            authority.value for authority in settings.authority_kinds
        ),
        "allowed_skill_ids": settings.allowed_skill_ids,
        "read_limits": settings.read_limits.as_model_dict(),
        "source_mappings": tuple(
            trusted_skill_source_identity_dict(source)
            for source in settings.sources
        ),
    }


def sanitize_server_protocol_value(
    value: object,
    *,
    tool_name: str | None = None,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
    protocol: ServerOutputRedactionProtocol = "openai",
    channel: ServerOutputRedactionChannel | None = None,
) -> object:
    """Return a protocol-safe value with configured redaction."""
    settings = coerce_server_output_redaction_settings(
        output_redaction_settings
    )
    return _sanitize_server_protocol_value(
        value,
        output_redaction_settings=settings,
        protocol=protocol,
        channel=channel,
        redact_skill_content=(
            _is_skills_tool_name(tool_name)
            and settings.should_redact(
                "skills_tool_content",
                protocol=protocol,
                channel=channel,
            )
        ),
    )


def sanitize_server_protocol_text(
    value: str,
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
    protocol: ServerOutputRedactionProtocol = "openai",
    channel: ServerOutputRedactionChannel | None = None,
) -> str:
    """Return protocol text with optional configured redaction."""
    assert isinstance(value, str)
    settings = coerce_server_output_redaction_settings(
        output_redaction_settings
    )
    if settings.should_redact(
        "host_paths",
        protocol=protocol,
        channel=channel,
    ):
        return _redact_host_paths(value)
    return value


def sanitize_model_visible_server_protocol_text(
    value: str,
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
    protocol: ServerOutputRedactionProtocol = "openai",
    channel: ServerOutputRedactionChannel = "answer",
) -> str:
    """Return model-visible text with configured redaction."""
    assert isinstance(value, str)
    settings = coerce_server_output_redaction_settings(
        output_redaction_settings
    )
    if _looks_like_echoed_skill_body(
        value,
        output_redaction_settings=settings,
        protocol=protocol,
        channel=channel,
    ):
        return SKILL_CONTENT_REDACTION
    if settings.should_redact(
        "host_paths",
        protocol=protocol,
        channel=channel,
    ):
        return _redact_host_paths(value)
    return value


class ModelVisibleServerProtocolTextRedactor:
    """Apply configured redaction to streaming model text."""

    def __init__(
        self,
        output_redaction_settings: ServerOutputRedactionSettings | None = None,
        *,
        protocol: ServerOutputRedactionProtocol = "openai",
        channel: ServerOutputRedactionChannel = "answer",
    ) -> None:
        self._settings = coerce_server_output_redaction_settings(
            output_redaction_settings
        )
        self._protocol = protocol
        self._channel = channel
        self._pending = ""
        self._redacted = False

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def push(self, value: str) -> tuple[str, ...]:
        """Return safe text chunks for a streaming delta."""
        assert isinstance(value, str)
        if not value or self._redacted:
            return ()
        candidate = self._pending + value
        self._pending = ""
        if self._should_buffer_skill_text_start() and (
            _could_be_echoed_skill_body_start_prefix(candidate)
        ):
            self._pending = candidate
            return ()
        if _has_echoed_skill_body_marker(
            candidate,
            output_redaction_settings=self._settings,
            protocol=self._protocol,
            channel=self._channel,
        ):
            self._redacted = True
            return (SKILL_CONTENT_REDACTION,)
        if self._should_redact_skill_source_paths() and (
            _should_buffer_skill_source_path_reference_candidate(candidate)
        ):
            self._pending = candidate
            return ()
        if self._should_redact_host_paths():
            safe_text, pending_tail = _split_streaming_host_path_tail(
                candidate
            )
        else:
            safe_text, pending_tail = candidate, ""
        if pending_tail:
            self._pending = pending_tail
            sanitized = self._sanitize_text(safe_text)
            return (sanitized,) if sanitized else ()
        if self._should_buffer_skill_body() and (
            _should_buffer_echoed_skill_body_candidate(candidate)
        ):
            self._pending = candidate
            return ()
        if self._should_buffer_skill_text_start() and (
            _should_hold_potential_skill_text_candidate(candidate)
        ):
            self._pending = candidate
            return ()
        sanitized = self._sanitize_text(candidate)
        return (sanitized,) if sanitized else ()

    def flush(self) -> tuple[str, ...]:
        """Return any pending safe text at stream completion."""
        if self._redacted or not self._pending:
            self._pending = ""
            return ()
        candidate = self._pending
        self._pending = ""
        sanitized = self._sanitize_text(candidate)
        return (sanitized,) if sanitized else ()

    def _sanitize_text(self, value: str) -> str:
        return sanitize_model_visible_server_protocol_text(
            value,
            output_redaction_settings=self._settings,
            protocol=self._protocol,
            channel=self._channel,
        )

    def _should_redact_host_paths(self) -> bool:
        return self._settings.should_redact(
            "host_paths",
            protocol=self._protocol,
            channel=self._channel,
        )

    def _should_buffer_skill_body(self) -> bool:
        return self._settings.should_redact(
            "skill_body_echoes",
            protocol=self._protocol,
            channel=self._channel,
        )

    def _should_redact_skill_source_paths(self) -> bool:
        return self._settings.should_redact(
            "skill_source_paths",
            protocol=self._protocol,
            channel=self._channel,
        )

    def _should_buffer_skill_text_start(self) -> bool:
        return self._should_buffer_skill_body() or (
            self._should_redact_skill_source_paths()
        )


def _sanitize_server_protocol_value(
    value: object,
    *,
    output_redaction_settings: ServerOutputRedactionSettings,
    protocol: ServerOutputRedactionProtocol,
    channel: ServerOutputRedactionChannel | None,
    redact_skill_content: bool,
) -> object:
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if redact_skill_content and key in _SKILL_CONTENT_KEYS:
                sanitized[key] = {
                    "redacted": True,
                    "reason": SKILL_CONTENT_REDACTION,
                }
                continue
            sanitized[key] = _sanitize_server_protocol_value(
                item,
                output_redaction_settings=output_redaction_settings,
                protocol=protocol,
                channel=channel,
                redact_skill_content=redact_skill_content,
            )
        return sanitized
    if isinstance(value, list | tuple):
        return [
            _sanitize_server_protocol_value(
                item,
                output_redaction_settings=output_redaction_settings,
                protocol=protocol,
                channel=channel,
                redact_skill_content=redact_skill_content,
            )
            for item in value
        ]
    if isinstance(value, str):
        return sanitize_server_protocol_text(
            value,
            output_redaction_settings=output_redaction_settings,
            protocol=protocol,
            channel=channel,
        )
    if isinstance(value, bytes | bytearray | memoryview):
        return "<redacted-bytes>"
    if isinstance(value, bool | int | float) or value is None:
        return value
    return sanitize_server_protocol_text(
        str(value),
        output_redaction_settings=output_redaction_settings,
        protocol=protocol,
        channel=channel,
    )


def _is_skills_tool_name(value: str | None) -> bool:
    return isinstance(value, str) and value.startswith(_SKILLS_TOOL_PREFIX)


def _redact_host_paths(value: str) -> str:
    return _HOST_PATH_PATTERN.sub(_redact_host_path_match, value)


def _split_streaming_host_path_tail(value: str) -> tuple[str, str]:
    start = _streaming_host_path_tail_start(value)
    if start is None:
        return value, ""
    return value[:start], value[start:]


def _streaming_host_path_tail_start(value: str) -> int | None:
    for index, char in enumerate(value):
        if char not in {"/", "\\"} and not (
            (index + 1 < len(value) and value[index + 1] == ":")
            or _could_be_streaming_windows_drive_prefix(value, index)
        ):
            continue
        tail = value[index:]
        if not _could_be_streaming_host_path_tail(tail):
            continue
        if _tail_inside_remote_url(value, index):
            continue
        return index
    return None


def _could_be_streaming_windows_drive_prefix(
    value: str,
    index: int,
) -> bool:
    if index != len(value) - 1:
        return False
    if index == 0 or not value[index - 1].isspace():
        return False
    return value[index].isupper() and value[index].isascii()


def _could_be_streaming_host_path_tail(value: str) -> bool:
    if not value or any(char.isspace() for char in value):
        return False
    return _could_be_streaming_posix_host_path_tail(
        value
    ) or _could_be_streaming_windows_host_path_tail(value)


def _could_be_streaming_posix_host_path_tail(value: str) -> bool:
    if not value.startswith("/"):
        return False
    return any(
        root.startswith(value) or value == root or value.startswith(f"{root}/")
        for root in _STREAM_HOST_PATH_POSIX_ROOTS
    )


def _could_be_streaming_windows_host_path_tail(value: str) -> bool:
    if len(value) == 1:
        return value.isupper() and value.isascii()
    if len(value) < 2 or value[1] != ":":
        return False
    if len(value) == 2:
        return True
    separator = value[2]
    if separator not in {"/", "\\"}:
        return False
    users_prefix = f"{separator}Users"
    suffix = value[2:]
    return (
        users_prefix.startswith(suffix)
        or suffix == users_prefix
        or suffix.startswith(f"{users_prefix}{separator}")
    )


def _tail_inside_remote_url(value: str, start: int) -> bool:
    token_start = start
    while token_start > 0 and not value[token_start - 1].isspace():
        token_start -= 1
    token_prefix = value[token_start:start]
    scheme_matches = tuple(_URL_SCHEME_PATTERN.finditer(token_prefix))
    if not scheme_matches:
        return False
    scheme_match = scheme_matches[-1]
    if scheme_match.group(0).lower() == "file:":
        return False
    return token_prefix[scheme_match.end() :].startswith("//")


def _path_match_inside_url(match: Match[str]) -> bool:
    start = match.start("path")
    token_start = start
    while token_start > 0 and not match.string[token_start - 1].isspace():
        token_start -= 1
    token_prefix = match.string[token_start:start]
    scheme_matches = tuple(_URL_SCHEME_PATTERN.finditer(token_prefix))
    if not scheme_matches:
        return False
    scheme_match = scheme_matches[-1]
    if scheme_match.group(0).lower() == "file:":
        return False
    return token_prefix[scheme_match.end() :].startswith("//")


def _looks_like_echoed_skill_body(
    value: str,
    *,
    output_redaction_settings: ServerOutputRedactionSettings,
    protocol: ServerOutputRedactionProtocol,
    channel: ServerOutputRedactionChannel,
) -> bool:
    stripped = value.strip()
    if output_redaction_settings.should_redact(
        "skill_source_paths",
        protocol=protocol,
        channel=channel,
    ) and _looks_like_skill_body_resource_reference(stripped):
        return True
    if len(stripped) < 40:
        return False
    return output_redaction_settings.should_redact(
        "skill_body_echoes",
        protocol=protocol,
        channel=channel,
    ) and _looks_like_skill_body_phrase_heading(stripped)


def _looks_like_skill_body_phrase_heading(value: str) -> bool:
    return any(
        _looks_like_skill_body_heading_title(match.group("title"))
        for match in _SKILL_BODY_HEADING_PATTERN.finditer(value)
    )


def _looks_like_skill_body_heading_title(value: str) -> bool:
    title = value.strip().strip("`")
    if not title:
        return False
    if _is_ordinary_title_heading(title):
        return False
    words = title.split()
    lowered_words = title.lower().split()
    if len(lowered_words) == 1:
        token = lowered_words[0]
        return (
            token == title
            or title.isupper()
            or title.istitle()
            or any(separator in token for separator in ":._-")
        ) and fullmatch(r"[a-z0-9][a-z0-9:._-]{0,63}", token) is not None
    if len(lowered_words) != 2:
        return False
    first, second = lowered_words
    return second == "skill" or (
        second == "basic"
        and words[0].isupper()
        and fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", first.lower()) is not None
    )


def _looks_like_skill_body_resource_reference(value: str) -> bool:
    return any(
        _looks_like_skill_body_resource_reference_candidate(
            value[match.start() :].lstrip()
        )
        for match in _SKILL_BODY_RESOURCE_START_PATTERN.finditer(value)
    )


def _looks_like_skill_body_resource_reference_candidate(value: str) -> bool:
    if _starts_with_ordinary_title_heading(value):
        return False
    bounded = _bounded_skill_body_resource_candidate(value)
    return contains_skill_source_resource_reference(
        bounded,
        SKILL_MAIN_RESOURCE_FILENAME,
    )


def _bounded_skill_body_resource_candidate(value: str) -> str:
    if value.startswith("#"):
        lines = value.split("\n", 1)
        if len(lines) == 1:
            return value
        body = _skill_heading_body_after_separator(value)
        return f"{lines[0]}\n\n{body[:_SKILL_BODY_DETECTION_WINDOW]}"
    lowered = value.lower()
    if lowered.startswith("description:"):
        prefix, separator, body = value.partition(":")
        return f"{prefix}{separator}{body[:_SKILL_BODY_DETECTION_WINDOW]}"
    return f"{value[:3]}{value[3:][:_SKILL_BODY_DETECTION_WINDOW]}"


def _starts_with_ordinary_title_heading(value: str) -> bool:
    stripped = value.lstrip()
    if not stripped.startswith("#"):
        return False
    heading = stripped.split("\n", 1)[0].lstrip("#").strip().strip("`")
    return _is_ordinary_title_heading(heading)


def _is_ordinary_title_heading(value: str) -> bool:
    return value.strip().strip("`").lower() in _ORDINARY_TITLE_HEADING_WORDS


def _has_echoed_skill_body_marker(
    value: str,
    *,
    output_redaction_settings: ServerOutputRedactionSettings,
    protocol: ServerOutputRedactionProtocol,
    channel: ServerOutputRedactionChannel,
) -> bool:
    return _looks_like_echoed_skill_body(
        value,
        output_redaction_settings=output_redaction_settings,
        protocol=protocol,
        channel=channel,
    )


def _could_be_echoed_skill_body_start_prefix(value: str) -> bool:
    tail = value.rsplit("\n", 1)[-1].lstrip()
    if not tail:
        return False
    lowered = tail.lower()
    return (
        "#".startswith(tail)
        or "---".startswith(tail)
        or "description:".startswith(lowered)
    )


def _should_buffer_echoed_skill_body_candidate(value: str) -> bool:
    matches = tuple(_SKILL_BODY_STREAM_START_PATTERN.finditer(value))
    if not matches:
        return False
    candidate = value[matches[-1].start() :].lstrip()
    lowered = candidate.lower()
    if candidate.startswith("#"):
        return _should_buffer_heading_skill_body_candidate(candidate)
    if lowered.startswith("description:"):
        return _should_buffer_skill_body_followup(candidate.split(":", 1)[1])
    after_marker = candidate[3:]
    return _should_buffer_skill_body_followup(after_marker)


def _should_hold_potential_skill_text_candidate(value: str) -> bool:
    matches = tuple(_SKILL_BODY_STREAM_START_PATTERN.finditer(value))
    if not matches:
        return False
    candidate = value[matches[-1].start() :].lstrip()
    if candidate.startswith("#"):
        lines = candidate.split("\n", 1)
        body = (
            _skill_heading_body_after_separator(candidate)
            if len(lines) > 1
            else ""
        )
        if len(body) > _SKILL_BODY_DETECTION_WINDOW:
            return False
        heading = lines[0].lstrip("#").strip()
        return _looks_like_definite_skill_heading_title(heading)
    return False


def _looks_like_definite_skill_heading_title(value: str) -> bool:
    title = value.strip().strip("`")
    if _is_ordinary_title_heading(title):
        return False
    return _looks_like_skill_body_heading_title(title)


def _should_buffer_skill_source_path_reference_candidate(
    value: str,
) -> bool:
    matches = tuple(_SKILL_BODY_STREAM_START_PATTERN.finditer(value))
    if not matches:
        return False
    candidate = value[matches[-1].start() :].lstrip()
    lowered = candidate.lower()
    if candidate.startswith("#"):
        lines = candidate.split("\n", 1)
        if len(lines) == 1:
            return True
        heading = lines[0].lstrip("#").strip()
        if _is_ordinary_title_heading(heading):
            return False
        body = _skill_heading_body_after_separator(candidate)
        if len(body) > _SKILL_BODY_DETECTION_WINDOW:
            return False
        return _should_buffer_skill_source_path_reference_followup(body)
    if lowered.startswith("description:"):
        return _should_buffer_skill_source_path_reference_followup(
            candidate.split(":", 1)[1]
        )
    after_marker = candidate[3:]
    return _should_buffer_skill_source_path_reference_followup(after_marker)


def _should_buffer_skill_source_path_reference_followup(
    value: str,
) -> bool:
    candidate = value.lstrip()
    if not candidate:
        return True
    lowered = candidate.lower()
    for marker in _SKILL_BODY_RESOURCE_FIELD_MARKERS:
        if marker.startswith(lowered):
            return True
        if lowered.startswith(marker):
            return could_contain_skill_source_path_reference_prefix(
                candidate.split(":", 1)[1]
            )
    return could_contain_skill_source_path_reference_prefix(candidate)


def _should_buffer_heading_skill_body_candidate(value: str) -> bool:
    lines = value.split("\n", 1)
    if len(lines) == 1:
        return True
    heading = lines[0].lstrip("#").strip()
    if _is_ordinary_title_heading(heading):
        return False
    body = _skill_heading_body_after_separator(value)
    return _should_buffer_skill_body_followup(body)


def _skill_heading_body_after_separator(value: str) -> str:
    lines = value.split("\n", 1)
    body = lines[1]
    match = _SKILL_BODY_HEADING_SEPARATOR_PATTERN.match(body)
    return body[match.end() :] if match else body


def _should_buffer_skill_body_followup(value: str) -> bool:
    candidate = value.lstrip()
    if not candidate:
        return True
    lowered = candidate.lower()
    if any(
        marker.startswith(lowered) or lowered.startswith(marker)
        for marker in _SKILL_BODY_TEXT_FOLLOWUP_MARKERS
    ):
        return True
    if could_contain_skill_source_path_reference_prefix(candidate):
        return True
    return _has_skill_body_marker_prefix_within_window(value)


def _has_skill_body_marker_prefix_within_window(value: str) -> bool:
    lowered = value.lower()
    max_marker_length = max(
        len(marker) for marker in _SKILL_BODY_TEXT_FOLLOWUP_MARKERS
    )
    suffix_start = max(0, len(lowered) - max_marker_length + 1)
    for start in range(suffix_start, len(lowered)):
        if start > _SKILL_BODY_DETECTION_WINDOW:
            continue
        suffix = lowered[start:]
        if not suffix or not _is_skill_body_marker_boundary(value, start):
            continue
        if any(
            marker.startswith(suffix)
            for marker in _SKILL_BODY_TEXT_FOLLOWUP_MARKERS
        ):
            return True
    return False


def _is_skill_body_marker_boundary(value: str, start: int) -> bool:
    if start <= 0:
        return True
    previous = value[start - 1]
    return not (previous.isalnum() or previous == "_")


def _redact_host_path_match(match: Match[str]) -> str:
    path = match.group("path")
    if _path_match_inside_url(match):
        return path
    name = PureWindowsPath(path).name if "\\" in path else Path(path).name
    return f"{HOST_PATH_REDACTION}/{name}" if name else HOST_PATH_REDACTION


class ResponseFormatText(BaseModel):
    type: Literal["text"]


class ResponseFormatJSONObject(BaseModel):
    type: Literal["json_object"]


class JSONSchemaField(BaseModel):
    title: str
    type: JSONType


class JSONSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    properties: dict[str, JSONSchemaField]
    required: list[str] | None = None
    title: str | None = None
    type: JSONType
    additionalProperties: bool | None = None


class JSONSchemaSettings(BaseModel):
    schema_: JSONSchema = Field(
        ..., validation_alias="schema", serialization_alias="schema"
    )
    name: str | None = None
    strict: bool = True


class ResponseFormatJSONSchema(BaseModel):
    type: Literal["json_schema"]
    json_schema: JSONSchemaSettings | None = None
    name: str | None = None
    schema_: JSONSchema | None = Field(
        None,
        validation_alias="schema",
        serialization_alias="schema",
    )
    strict: bool | None = None

    @model_validator(mode="after")
    def validate_schema_shape(self) -> "ResponseFormatJSONSchema":
        has_chat_schema = self.json_schema is not None
        has_responses_schema = self.schema_ is not None
        if has_chat_schema == has_responses_schema:
            raise ValueError(
                "Provide either json_schema or schema for json_schema format"
            )
        if has_chat_schema and (
            self.name is not None or self.strict is not None
        ):
            raise ValueError(
                "Chat-style json_schema format cannot include Responses fields"
            )
        if has_responses_schema and self.name is not None and not self.name:
            raise ValueError("Responses json_schema name cannot be empty")
        return self


class FunctionParameters(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["object"] = "object"
    properties: dict[str, JSONSchemaField]
    required: list[str] | None = None


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: FunctionParameters

    @model_validator(mode="after")
    def validate_remote_skills_tool_definition(
        self,
    ) -> "FunctionDefinition":
        if _is_skills_tool_name(self.name):
            raise ValueError("Remote requests cannot define skills tools")
        return self


class ToolFunction(BaseModel):
    type: Literal["function"]
    function: FunctionDefinition


class ContentText(BaseModel):
    type: Literal["text", "input_text"]
    text: str


class ContentImage(BaseModel):
    type: Literal["image_url"]
    image_url: dict[str, str]

    @model_validator(mode="after")
    def validate_remote_runtime_authority(self) -> "ContentImage":
        reject_remote_runtime_authority_fields(
            self.image_url,
            path="image_url",
        )
        return self


class ContentFile(BaseModel):
    type: Literal["file", "input_file"]
    file: dict[str, Any] | None = None
    file_data: str | None = None
    file_id: str | None = None
    file_url: str | None = None
    filename: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "ContentFile":
        nested = self.file or {}
        reject_remote_runtime_authority_fields(nested, path="file")
        has_source = any(
            _has_non_empty_file_source(value)
            for value in (
                (
                    nested.get("file_data")
                    if "file_data" in nested
                    else nested.get("data")
                ),
                nested.get("file_id"),
                (
                    nested.get("file_url")
                    if "file_url" in nested
                    else nested.get("url")
                ),
                self.file_data,
                self.file_id,
                self.file_url,
            )
        )
        if not has_source:
            raise ValueError(
                "File content requires file_id, file_url, file_data, or file"
            )
        return self


ResponseFormat = Annotated[
    ResponseFormatText | ResponseFormatJSONObject | ResponseFormatJSONSchema,
    Field(discriminator="type"),
]

Tool = Annotated[ToolFunction, Field(discriminator="type")]

ContentPart = Annotated[
    ContentText | ContentImage | ContentFile, Field(discriminator="type")
]


class ChatMessage(BaseModel):
    role: MessageRole
    content: str | list[ContentPart]


class ReasoningConfig(BaseModel):
    effort: ReasoningEffort | None = None


class ResponsesTextConfig(BaseModel):
    format: ResponseFormat | None = None
    stop: str | list[str] | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = Field(
        None,
        description=(
            "ID of the model to use for generating the completion. When"
            " omitted, use the server's configured model."
        ),
    )
    messages: list[ChatMessage] = Field(
        ..., description="List of messages in the conversation"
    )
    temperature: float | None = Field(
        1.0, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float | None = Field(
        1.0, ge=0.0, le=1.0, description="Nucleus sampling probability"
    )
    n: int | None = Field(
        1, ge=1, description="Number of completions to generate"
    )
    stream: bool | None = Field(
        False, description="Whether to stream back partial progress"
    )
    stop: str | list[str] | None = Field(
        None,
        description=(
            "Sequence where the API will stop generating further tokens"
        ),
    )
    max_tokens: int | None = Field(
        None, ge=1, description="Maximum tokens to generate in the completion"
    )
    presence_penalty: float | None = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Penalty for new tokens based on whether they appear in text"
            " so far"
        ),
    )
    frequency_penalty: float | None = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Penalty for new tokens based on their frequency in text so far"
        ),
    )
    logit_bias: dict[str, int] | None = Field(
        None,
        description=(
            "Modify the likelihood of specified tokens appearing in the"
            " completion"
        ),
    )
    logprobs: bool | None = None
    top_logprobs: int | None = Field(None, ge=0, le=5)
    user: str | None = Field(
        None, description="Unique identifier representing your end-user"
    )
    response_format: ResponseFormat | None = Field(
        None, description="Format to use for model response"
    )
    reasoning_effort: ReasoningEffort | None = Field(
        None,
        description="Reasoning effort for supported reasoning models",
    )
    tools: list[Tool] | None = None
    tool_choice: (
        Literal["auto", "none", "required"] | str | dict[str, object] | None
    ) = None

    @model_validator(mode="before")
    @classmethod
    def validate_remote_runtime_authority(cls, value: object) -> object:
        _reject_request_remote_runtime_authority(
            value,
            allowed_fields=frozenset(cls.model_fields),
            path="chat",
        )
        if isinstance(value, Mapping):
            reject_remote_runtime_authority_model_identifier(
                value.get("model"),
                path="chat.model",
            )
            reject_remote_runtime_authority_fields(
                value.get("tool_choice"),
                path="chat.tool_choice",
            )
        return value


ResponsesInput = str | list[ChatMessage]


class ResponsesRequest(BaseModel):
    model: str | None = Field(
        None,
        description=(
            "ID of the model to use for generating the response. When"
            " omitted, use the server's configured model."
        ),
    )
    instructions: str | None = None
    input: ResponsesInput = Field(...)
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    text: ResponsesTextConfig | None = None
    response_format: ResponseFormat | None = None
    reasoning: ReasoningConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_remote_runtime_authority(cls, value: object) -> object:
        _reject_request_remote_runtime_authority(
            value,
            allowed_fields=frozenset(cls.model_fields),
            path="responses",
        )
        if isinstance(value, Mapping):
            reject_remote_runtime_authority_model_identifier(
                value.get("model"),
                path="responses.model",
            )
            _reject_remote_skills_tool_definitions(
                value.get("tools"),
                path="responses.tools",
            )
        return value

    @property
    def messages(self) -> list[ChatMessage]:
        if isinstance(self.input, str):
            return [ChatMessage(role=MessageRole.USER, content=self.input)]
        return self.input

    @model_validator(mode="after")
    def validate_text_aliases(self) -> "ResponsesRequest":
        if (
            self.text is not None
            and self.text.format is not None
            and self.response_format is not None
        ):
            raise ValueError("Use either text.format or response_format")
        if (
            self.text is not None
            and self.text.stop is not None
            and self.stop is not None
        ):
            raise ValueError("Use either text.stop or stop")
        return self


def _reject_request_remote_runtime_authority(
    value: object,
    *,
    allowed_fields: frozenset[str],
    path: str,
) -> None:
    reject_remote_runtime_authority_extra_fields(
        value,
        allowed_fields=allowed_fields,
        allow_container_profile_selector=True,
        path=path,
    )


def _reject_remote_skills_tool_definitions(
    value: object,
    *,
    path: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, list | tuple):
        return
    for index, tool in enumerate(value):
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if _is_skills_tool_name(name if isinstance(name, str) else None):
            raise ValueError(
                "Remote requests cannot define skills tools"
                f" at {path}[{index}].function.name"
            )


class MCPFileDescriptor(BaseModel):
    model_config = ConfigDict(
        json_schema_extra=_mcp_file_descriptor_json_schema
    )

    file_data: str | None = Field(
        None,
        alias="data",
        description="Base64 file contents. Aliases: data, base64, file_data.",
    )
    file_url: str | None = Field(
        None,
        alias="uri",
        description="File URI or URL. Aliases: uri, url, file_url.",
    )
    mime_type: str | None = Field(
        None,
        alias="mimeType",
        description="File MIME type. Aliases: mimeType, mime_type.",
    )
    filename: str | None = Field(
        None, description="Optional file name to send with the file."
    )

    @model_validator(mode="before")
    @classmethod
    def validate_descriptor_input(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("File descriptor must be an object")
        source_keys = _present_file_source_keys(value)
        if len(source_keys) != 1:
            raise ValueError(
                "File descriptor requires exactly one file source"
            )
        source = value[source_keys[0]]
        if not _has_non_empty_file_source(source):
            raise ValueError(
                "File descriptor source must be a non-empty string"
            )
        if source_keys[0] in MCP_FILE_DATA_KEYS:
            source = _validate_base64_file_source(source)
        else:
            source = source.strip()
        mime_type_keys = _present_keys(value, MCP_FILE_MIME_TYPE_KEYS)
        if len(mime_type_keys) > 1:
            raise ValueError("File descriptor must use one MIME type key")

        normalized = dict(value)
        for key in MCP_FILE_SOURCE_KEYS:
            normalized.pop(key, None)
        if source_keys[0] in MCP_FILE_DATA_KEYS:
            normalized["data"] = source
        else:
            normalized["uri"] = source

        if mime_type_keys:
            mime_type = value[mime_type_keys[0]]
            if isinstance(mime_type, str):
                mime_type = mime_type.strip()
            for key in MCP_FILE_MIME_TYPE_KEYS:
                normalized.pop(key, None)
            normalized["mimeType"] = mime_type

        filename_keys = _present_keys(value, MCP_FILE_FILENAME_KEYS)
        if len(filename_keys) > 1:
            raise ValueError("File descriptor must use one filename key")
        if filename_keys:
            filename = value[filename_keys[0]]
            if isinstance(filename, str):
                filename = filename.strip()
            for key in MCP_FILE_FILENAME_KEYS:
                normalized.pop(key, None)
            normalized["filename"] = filename
        filename = normalized.get("filename")
        if isinstance(filename, str):
            normalized["filename"] = filename.strip()

        return normalized

    @model_validator(mode="after")
    def validate_sources(self) -> "MCPFileDescriptor":
        has_file_data = _has_non_empty_file_source(self.file_data)
        has_file_url = _has_non_empty_file_source(self.file_url)
        if has_file_data == has_file_url:
            raise ValueError(
                "File descriptor requires exactly one file source"
            )
        if self.mime_type is not None and not self.mime_type.strip():
            raise ValueError("File descriptor mime_type cannot be empty")
        if self.filename is not None and not self.filename.strip():
            raise ValueError("File descriptor filename cannot be empty")
        return self

    def as_content_file(self) -> dict[str, str]:
        payload: dict[str, str] = {}
        if self.file_data is not None:
            payload["file_data"] = self.file_data
        if self.file_url is not None:
            payload["file_url"] = self.file_url
        if self.mime_type is not None:
            payload["mime_type"] = self.mime_type
        if self.filename is not None:
            payload["filename"] = self.filename
        return payload


class MCPToolRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra=_mcp_tool_request_json_schema)

    input_string: str | None = Field(
        None, description="Input to pass to the orchestrator via MCP"
    )
    files: list[MCPFileDescriptor] = Field(
        default_factory=list,
        description="JSON file descriptors to attach to the input.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_request_input(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        file_keys = _present_keys(
            value, ("files", "input_files", "file_descriptors")
        )
        if len(file_keys) > 1:
            raise ValueError("MCP request must use one files key")
        if not file_keys or file_keys[0] == "files":
            return value

        normalized = dict(value)
        normalized["files"] = normalized.pop(file_keys[0])
        return normalized

    @model_validator(mode="after")
    def validate_input(self) -> "MCPToolRequest":
        if not self.files and (
            self.input_string is None or not self.input_string.strip()
        ):
            raise ValueError("MCP request requires input_string or files")
        return self


class ChatCompletionChunkChoiceDelta(BaseModel):
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkChoiceDelta


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: list[dict[str, Any]]


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class EngineRequest(BaseModel):
    uri: str | None = None
    database: str | None = None

    @model_validator(mode="after")
    def check_uri_or_database(self) -> "EngineRequest":
        if self.uri is None and self.database is None:
            raise ValueError("Provide uri or database")
        return self
