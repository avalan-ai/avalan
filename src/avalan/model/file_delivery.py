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
class FileDeliveryRequest:
    mime_type: str | None = None
    size_bytes: int | None = None
    has_artifact: bool = False
    metadata: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if self.mime_type is not None:
            _assert_non_empty_string(self.mime_type, "mime_type")
        _assert_optional_non_negative_int(self.size_bytes, "size_bytes")
        assert isinstance(self.has_artifact, bool)
        _assert_metadata(self.metadata)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FileDeliveryDecision:
    mode: FileDeliveryMode
    reference: str | None = None
    diagnostic: FileDeliveryDiagnostic | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.mode, FileDeliveryMode)
        if self.reference is not None:
            _assert_non_empty_string(self.reference, "reference")
        if self.diagnostic is not None:
            assert isinstance(self.diagnostic, FileDeliveryDiagnostic)

    @property
    def needs_artifact_read(self) -> bool:
        return self.mode in {
            FileDeliveryMode.INLINE_BYTES,
            FileDeliveryMode.INLINE_TEXT,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class _ProviderReference:
    kind: str
    provider: str
    reference: str


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
            "provider_reference",
            "remote_url",
        }
    )
    object_store_uri_schemes: tuple[str, ...] = ()
    limits: tuple[FileDeliveryLimit, ...] = ()
    inline_byte_limit: FileDeliveryLimit | None = None
    inline_text_limit: FileDeliveryLimit | None = None
    file_count_limit: FileDeliveryLimit | None = None
    requires_conversion_for_file_blocks: bool = False
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
        for field_name in (
            "inline_byte_limit",
            "inline_text_limit",
            "file_count_limit",
        ):
            limit = getattr(self, field_name)
            if limit is not None:
                assert isinstance(
                    limit, FileDeliveryLimit
                ), f"{field_name} must be a FileDeliveryLimit"
        assert isinstance(
            self.requires_conversion_for_file_blocks, bool
        ), "requires_conversion_for_file_blocks must be a bool"
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
    def supports_file_delivery(self) -> bool:
        return bool(self.delivery_modes - frozenset({FileDeliveryMode.REJECT}))

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

    @property
    def has_reference_delivery(self) -> bool:
        return bool(
            self.delivery_modes
            & frozenset(
                {
                    FileDeliveryMode.PROVIDER_FILE_ID,
                    FileDeliveryMode.HOSTED_URL,
                    FileDeliveryMode.OBJECT_STORE_URI,
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

    def accepts_file_count(self, count: int) -> bool:
        assert isinstance(count, int), "count must be an integer"
        assert not isinstance(count, bool), "count must be an integer"
        assert count >= 0, "count must be non-negative"
        if self.file_count_limit is None:
            return True
        if self.file_count_limit.max_count is None:
            return True
        return count <= self.file_count_limit.max_count

    def plan_delivery(
        self,
        request: FileDeliveryRequest,
    ) -> FileDeliveryDecision:
        return plan_file_delivery(self, request)


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


def plan_file_delivery(
    profile: FileDeliveryProfile,
    request: FileDeliveryRequest,
) -> FileDeliveryDecision:
    assert isinstance(profile, FileDeliveryProfile)
    assert isinstance(request, FileDeliveryRequest)
    if not profile.supports_file_delivery:
        return _reject(
            code="model.file_delivery.unsupported",
            message="Model file delivery is not supported.",
            hint="Use a target with file support or add a conversion path.",
        )
    source_kind = _request_source_kind(request)
    if source_kind is not None and not profile.accepts_source_kind(
        source_kind
    ):
        return _reject(
            code="model.file_delivery.unsupported_source_kind",
            message="Model file delivery does not accept this source kind.",
            hint=(
                "Use a file source kind accepted by the target profile or "
                "declare conversion."
            ),
        )
    if request.mime_type is not None and not profile.accepts_mime_type(
        request.mime_type
    ):
        return _reject(
            code="model.file_delivery.unsupported_mime_type",
            message="Model file delivery does not accept this MIME type.",
            hint=(
                "Use an accepted MIME type or convert the file before"
                " dispatch."
            ),
        )

    provider_reference = _provider_reference(request.metadata)
    if provider_reference is not None:
        decision = _plan_provider_reference(profile, provider_reference)
        if decision is not None:
            return decision

    if request.has_artifact and _can_inline_bytes(profile, request):
        limit_diagnostic = _limit_diagnostic(
            request,
            profile.inline_byte_limit,
            expands_base64=True,
        )
        if limit_diagnostic is not None:
            return limit_diagnostic
        return FileDeliveryDecision(mode=FileDeliveryMode.INLINE_BYTES)

    if request.has_artifact and _can_inline_text(profile, request):
        limit_diagnostic = _limit_diagnostic(
            request,
            profile.inline_text_limit,
        )
        if limit_diagnostic is not None:
            return limit_diagnostic
        return FileDeliveryDecision(mode=FileDeliveryMode.INLINE_TEXT)

    return _reject(
        code="model.file_delivery.no_supported_delivery_mode",
        message="Model file delivery has no supported delivery mode.",
        hint="Provide a compatible file reference, artifact, or conversion.",
    )


def _plan_provider_reference(
    profile: FileDeliveryProfile,
    reference: _ProviderReference,
) -> FileDeliveryDecision | None:
    if reference.provider != profile.name:
        return _reject(
            code="model.file_delivery.provider_mismatch",
            message="Model file delivery reference targets another provider.",
            hint="Use a file reference owned by the selected target provider.",
        )
    match reference.kind:
        case "provider_file_id":
            if profile.supports_delivery_mode(
                FileDeliveryMode.PROVIDER_FILE_ID
            ):
                return FileDeliveryDecision(
                    mode=FileDeliveryMode.PROVIDER_FILE_ID,
                    reference=reference.reference,
                )
        case "hosted_url" | "expiring_provider_handle":
            if profile.supports_delivery_mode(FileDeliveryMode.HOSTED_URL):
                return FileDeliveryDecision(
                    mode=FileDeliveryMode.HOSTED_URL,
                    reference=reference.reference,
                )
        case "object_store_uri":
            if profile.supports_delivery_mode(
                FileDeliveryMode.OBJECT_STORE_URI
            ) and profile.allows_object_store_uri(reference.reference):
                return FileDeliveryDecision(
                    mode=FileDeliveryMode.OBJECT_STORE_URI,
                    reference=reference.reference,
                )
            return _reject(
                code="model.file_delivery.unsupported_object_store_uri",
                message="Model file delivery does not accept this object URI.",
                hint=(
                    "Use an object URI scheme accepted by the target profile."
                ),
            )
    return _reject(
        code="model.file_delivery.unsupported_provider_reference",
        message="Model file delivery does not accept this provider reference.",
        hint="Use a provider reference kind accepted by the target profile.",
    )


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
        inline_byte_limit=_inline_byte_limit(provider),
        inline_text_limit=_inline_text_limit(provider),
        file_count_limit=_file_count_limit(provider),
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
        inline_byte_limit=_inline_byte_limit("bedrock"),
        inline_text_limit=_inline_text_limit("bedrock"),
        file_count_limit=_file_count_limit("bedrock"),
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
        inline_text_limit=_inline_text_limit("local_text"),
        file_count_limit=_file_count_limit("local_text"),
        requires_conversion_for_file_blocks=True,
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
            "video/*",
        ),
        object_store_uri_schemes=(),
        limits=_common_limits("local_multimodal"),
        inline_byte_limit=_inline_byte_limit("local_multimodal"),
        inline_text_limit=_inline_text_limit("local_multimodal"),
        file_count_limit=_file_count_limit("local_multimodal"),
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


def _inline_byte_limit(provider: str) -> FileDeliveryLimit:
    return FileDeliveryLimit(
        name="inline_file_bytes",
        source=f"provider.{provider}",
    )


def _inline_text_limit(provider: str) -> FileDeliveryLimit:
    return FileDeliveryLimit(
        name="inline_text_bytes",
        source=f"provider.{provider}",
    )


def _file_count_limit(provider: str) -> FileDeliveryLimit:
    return FileDeliveryLimit(
        name="file_count",
        source=f"provider.{provider}",
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


def _can_inline_bytes(
    profile: FileDeliveryProfile,
    request: FileDeliveryRequest,
) -> bool:
    if not profile.supports_delivery_mode(FileDeliveryMode.INLINE_BYTES):
        return False
    return _has_provider_file_reference(profile) or _can_inline_text(
        profile,
        request,
    )


def _has_provider_file_reference(profile: FileDeliveryProfile) -> bool:
    return profile.supports_delivery_mode(
        FileDeliveryMode.PROVIDER_FILE_ID
    ) or profile.supports_delivery_mode(FileDeliveryMode.HOSTED_URL)


def _can_inline_text(
    profile: FileDeliveryProfile,
    request: FileDeliveryRequest,
) -> bool:
    return profile.supports_delivery_mode(
        FileDeliveryMode.INLINE_TEXT
    ) and profile.accepts_mime_type(request.mime_type)


def _limit_diagnostic(
    request: FileDeliveryRequest,
    limit: FileDeliveryLimit | None,
    *,
    expands_base64: bool = False,
) -> FileDeliveryDecision | None:
    assert isinstance(expands_base64, bool)
    if limit is None or limit.max_bytes is None or request.size_bytes is None:
        return None
    delivery_size = (
        _base64_size(request.size_bytes)
        if expands_base64
        else request.size_bytes
    )
    if delivery_size <= limit.max_bytes:
        return None
    return _reject(
        code="model.file_delivery.inline_limit_exceeded",
        message="Model file delivery inline limit would be exceeded.",
        hint=f"Use a delivery mode within the {limit.name} limit.",
    )


def _base64_size(size_bytes: int) -> int:
    assert isinstance(size_bytes, int)
    assert not isinstance(size_bytes, bool)
    assert size_bytes >= 0
    return ((size_bytes + 2) // 3) * 4


def _reject(
    *,
    code: str,
    message: str,
    hint: str,
) -> FileDeliveryDecision:
    return FileDeliveryDecision(
        mode=FileDeliveryMode.REJECT,
        diagnostic=FileDeliveryDiagnostic(
            code=code,
            message=message,
            hint=hint,
        ),
    )


def _metadata_string(
    metadata: Mapping[str, object],
    key: str,
) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _request_source_kind(request: FileDeliveryRequest) -> str | None:
    source_kind = _metadata_string(request.metadata, "source_kind")
    if source_kind is not None:
        return source_kind
    if _has_reference_metadata(request.metadata):
        return "provider_reference"
    if request.has_artifact:
        return "artifact"
    return None


def _has_reference_metadata(metadata: Mapping[str, object]) -> bool:
    return "provider_reference" in metadata


def _provider_reference(
    metadata: Mapping[str, object],
) -> _ProviderReference | None:
    value = metadata.get("provider_reference")
    if not isinstance(value, Mapping):
        return None
    kind = value.get("kind")
    provider = value.get("provider")
    reference = value.get("reference")
    if (
        not isinstance(kind, str)
        or not kind.strip()
        or not isinstance(provider, str)
        or not provider.strip()
        or not isinstance(reference, str)
        or not reference.strip()
    ):
        return None
    return _ProviderReference(
        kind=kind,
        provider=provider,
        reference=reference,
    )


def _mime_type_matches(mime_type: str, pattern: str) -> bool:
    if pattern == "*/*":
        return True
    if pattern.endswith("/*"):
        return mime_type.startswith(pattern[:-1])
    return mime_type == pattern


def _assert_non_empty_string(value: str, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value, f"{field_name} must not be empty"


def _assert_optional_non_negative_int(
    value: int | None,
    field_name: str,
) -> None:
    if value is None:
        return
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    assert value >= 0, f"{field_name} must be non-negative"


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


def _assert_metadata(metadata: Mapping[str, object]) -> None:
    assert isinstance(metadata, Mapping), "metadata must be a mapping"
    for key in metadata:
        _assert_non_empty_string(key, "metadata keys")


_TEXT_MIME_PATTERNS = (
    "text/*",
    "application/json",
    "application/markdown",
    "application/xml",
)
