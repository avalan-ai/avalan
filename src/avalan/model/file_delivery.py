from ..entities import Vendor

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import get_args
from urllib.parse import urlparse


class FileDeliveryMode(StrEnum):
    PROVIDER_FILE_ID = "provider_file_id"
    HOSTED_URL = "hosted_url"
    OBJECT_STORE_URI = "object_store_uri"
    INLINE_BYTES = "inline_bytes"
    INLINE_TEXT = "inline_text"
    CONVERTED_ARTIFACT = "converted_artifact"
    RETRIEVAL_CONTEXT = "retrieval_context"
    MAP_REDUCE_CONTEXT = "map_reduce_context"
    REJECT = "reject"


class LocalFileDeliveryProfile(StrEnum):
    TEXT = "text"
    MULTIMODAL = "multimodal"


@dataclass(frozen=True, slots=True, kw_only=True)
class FileDeliveryLimit:
    name: str
    source: str
    max_bytes: int | None = None
    max_count: int | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.name, "name")
        _assert_non_empty_string(self.source, "source")
        _assert_optional_positive_int(self.max_bytes, "max_bytes")
        _assert_optional_positive_int(self.max_count, "max_count")
        _assert_optional_positive_int(self.max_tokens, "max_tokens")


@dataclass(frozen=True, slots=True, kw_only=True)
class FileDeliveryDiagnostic:
    code: str
    message: str
    hint: str

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.code, "code")
        _assert_non_empty_string(self.message, "message")
        _assert_non_empty_string(self.hint, "hint")


@dataclass(frozen=True, slots=True, kw_only=True)
class FileDeliveryProfile:
    name: str
    delivery_modes: frozenset[FileDeliveryMode]
    accepted_mime_types: tuple[str, ...] = ("*/*",)
    source_kinds: frozenset[str] = frozenset(
        {
            "artifact",
            "inline_bytes",
            "local_path",
            "remote_url",
        }
    )
    object_store_uri_schemes: tuple[str, ...] = ()
    limits: tuple[FileDeliveryLimit, ...] = ()
    diagnostics: tuple[FileDeliveryDiagnostic, ...] = ()
    metadata: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.name, "name")
        assert isinstance(
            self.delivery_modes, frozenset
        ), "delivery_modes must be a frozenset"
        for mode in self.delivery_modes:
            assert isinstance(
                mode, FileDeliveryMode
            ), "delivery_modes must contain FileDeliveryMode values"
        _assert_string_tuple(self.accepted_mime_types, "accepted_mime_types")
        assert isinstance(
            self.source_kinds, frozenset
        ), "source_kinds must be a frozenset"
        for source_kind in self.source_kinds:
            _assert_non_empty_string(source_kind, "source_kinds")
        _assert_string_tuple(
            self.object_store_uri_schemes,
            "object_store_uri_schemes",
        )
        for limit in self.limits:
            assert isinstance(
                limit, FileDeliveryLimit
            ), "limits must contain FileDeliveryLimit values"
        for diagnostic in self.diagnostics:
            assert isinstance(
                diagnostic, FileDeliveryDiagnostic
            ), "diagnostics must contain FileDeliveryDiagnostic values"
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @property
    def has_native_file_delivery(self) -> bool:
        return bool(
            self.delivery_modes
            & frozenset(
                {
                    FileDeliveryMode.PROVIDER_FILE_ID,
                    FileDeliveryMode.HOSTED_URL,
                    FileDeliveryMode.OBJECT_STORE_URI,
                    FileDeliveryMode.INLINE_BYTES,
                }
            )
        )

    def supports_delivery_mode(self, mode: FileDeliveryMode) -> bool:
        assert isinstance(mode, FileDeliveryMode)
        return mode in self.delivery_modes

    def accepts_mime_type(self, mime_type: str | None) -> bool:
        if mime_type is None:
            return False
        return any(
            _mime_type_matches(mime_type, pattern)
            for pattern in self.accepted_mime_types
        )

    def accepts_source_kind(self, source_kind: str) -> bool:
        _assert_non_empty_string(source_kind, "source_kind")
        return source_kind in self.source_kinds

    def allows_object_store_uri(self, uri: str) -> bool:
        _assert_non_empty_string(uri, "uri")
        scheme = urlparse(uri).scheme.lower()
        return bool(scheme) and scheme in self.object_store_uri_schemes


def resolve_file_delivery_profile(
    uri: str | None,
    *,
    local_profile: LocalFileDeliveryProfile = LocalFileDeliveryProfile.TEXT,
) -> FileDeliveryProfile:
    assert isinstance(local_profile, LocalFileDeliveryProfile)
    provider = _provider_from_uri(uri)
    if provider == "openai":
        return _hosted_provider_profile("openai")
    if provider == "anthropic":
        return _hosted_provider_profile("anthropic")
    if provider == "google":
        return _hosted_provider_profile(
            "google",
            object_store_uri_schemes=("gs",),
        )
    if provider == "bedrock":
        return _bedrock_profile()
    if provider == "local":
        return (
            _local_multimodal_profile()
            if (local_profile == LocalFileDeliveryProfile.MULTIMODAL)
            else _local_text_profile()
        )
    return _unknown_profile()


def _hosted_provider_profile(
    provider: str,
    *,
    object_store_uri_schemes: tuple[str, ...] = (),
) -> FileDeliveryProfile:
    modes = {
        FileDeliveryMode.PROVIDER_FILE_ID,
        FileDeliveryMode.HOSTED_URL,
        FileDeliveryMode.INLINE_BYTES,
        FileDeliveryMode.INLINE_TEXT,
        FileDeliveryMode.CONVERTED_ARTIFACT,
        FileDeliveryMode.RETRIEVAL_CONTEXT,
        FileDeliveryMode.MAP_REDUCE_CONTEXT,
    }
    if object_store_uri_schemes:
        modes.add(FileDeliveryMode.OBJECT_STORE_URI)
    return FileDeliveryProfile(
        name=provider,
        delivery_modes=frozenset(modes),
        object_store_uri_schemes=object_store_uri_schemes,
        limits=_common_limits(provider),
    )


def _bedrock_profile() -> FileDeliveryProfile:
    return FileDeliveryProfile(
        name="bedrock",
        delivery_modes=frozenset(
            {
                FileDeliveryMode.OBJECT_STORE_URI,
                FileDeliveryMode.INLINE_BYTES,
                FileDeliveryMode.INLINE_TEXT,
                FileDeliveryMode.CONVERTED_ARTIFACT,
                FileDeliveryMode.RETRIEVAL_CONTEXT,
                FileDeliveryMode.MAP_REDUCE_CONTEXT,
            }
        ),
        accepted_mime_types=_TEXT_MIME_PATTERNS,
        object_store_uri_schemes=("s3",),
        limits=_common_limits("bedrock"),
    )


def _local_text_profile() -> FileDeliveryProfile:
    return FileDeliveryProfile(
        name="local_text",
        delivery_modes=frozenset(
            {
                FileDeliveryMode.INLINE_TEXT,
                FileDeliveryMode.CONVERTED_ARTIFACT,
                FileDeliveryMode.RETRIEVAL_CONTEXT,
                FileDeliveryMode.MAP_REDUCE_CONTEXT,
            }
        ),
        accepted_mime_types=_TEXT_MIME_PATTERNS,
        object_store_uri_schemes=(),
        limits=_common_limits("local_text"),
    )


def _local_multimodal_profile() -> FileDeliveryProfile:
    return FileDeliveryProfile(
        name="local_multimodal",
        delivery_modes=frozenset(
            {
                FileDeliveryMode.INLINE_BYTES,
                FileDeliveryMode.INLINE_TEXT,
                FileDeliveryMode.CONVERTED_ARTIFACT,
                FileDeliveryMode.RETRIEVAL_CONTEXT,
                FileDeliveryMode.MAP_REDUCE_CONTEXT,
            }
        ),
        accepted_mime_types=(
            "audio/*",
            "image/*",
            "text/*",
            "video/*",
            "application/json",
            "application/markdown",
            "application/xml",
        ),
        object_store_uri_schemes=(),
        limits=_common_limits("local_multimodal"),
    )


def _unknown_profile() -> FileDeliveryProfile:
    return FileDeliveryProfile(
        name="unknown",
        delivery_modes=frozenset({FileDeliveryMode.REJECT}),
        accepted_mime_types=(),
        source_kinds=frozenset(),
        limits=(),
        diagnostics=(
            FileDeliveryDiagnostic(
                code="model.file_delivery.unknown_profile",
                message="Model file delivery capabilities are unknown.",
                hint=(
                    "Configure an explicit provider or backend file delivery "
                    "profile before sending file inputs."
                ),
            ),
        ),
    )


def _common_limits(provider: str) -> tuple[FileDeliveryLimit, ...]:
    return (
        FileDeliveryLimit(name="task_file_bytes", source="task.limits"),
        FileDeliveryLimit(name="artifact_bytes", source="task.artifact"),
        FileDeliveryLimit(
            name="provider_file_bytes",
            source=f"provider.{provider}",
        ),
        FileDeliveryLimit(
            name="model_context_tokens",
            source="model.context",
        ),
        FileDeliveryLimit(name="conversion_bytes", source="converter"),
    )


def _provider_from_uri(uri: str | None) -> str | None:
    if not isinstance(uri, str) or not uri.startswith("ai://"):
        return None
    parsed = urlparse(uri)
    provider = parsed.hostname
    if provider == "local":
        return "local"
    if provider in get_args(Vendor):
        return provider
    return None


def _mime_type_matches(mime_type: str, pattern: str) -> bool:
    if pattern == "*/*":
        return True
    if pattern.endswith("/*"):
        return mime_type.startswith(pattern[:-1])
    return mime_type == pattern


def _assert_non_empty_string(value: str, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value, f"{field_name} must not be empty"


def _assert_optional_positive_int(
    value: int | None,
    field_name: str,
) -> None:
    if value is None:
        return
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    assert value > 0, f"{field_name} must be positive"


def _assert_string_tuple(values: tuple[str, ...], field_name: str) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        _assert_non_empty_string(value, field_name)


_TEXT_MIME_PATTERNS = (
    "text/*",
    "application/json",
    "application/markdown",
    "application/xml",
)
