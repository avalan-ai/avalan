from ..types import assert_non_empty_string as _assert_non_empty_string
from .store import freeze_snapshot_metadata

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from types import MappingProxyType

_REDACTED_METADATA_SUMMARY = MappingProxyType({"privacy": "<redacted>"})


class TaskFileSourceKind(StrEnum):
    LOCAL_PATH = "local_path"
    REMOTE_URL = "remote_url"
    ARTIFACT = "artifact"
    INLINE_BYTES = "inline_bytes"
    PROVIDER_REFERENCE = "provider_reference"


class TaskProviderReferenceKind(StrEnum):
    PROVIDER_FILE_ID = "provider_file_id"
    HOSTED_URL = "hosted_url"
    OBJECT_STORE_URI = "object_store_uri"
    EXPIRING_PROVIDER_HANDLE = "expiring_provider_handle"


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskProviderReference:
    kind: TaskProviderReferenceKind
    provider: str
    reference: str
    owner_scope: str | None = None
    expires_at: datetime | None = None
    mime_type: str | None = None
    size_bucket: str | None = None
    identity_hmac: str | None = None
    durable: bool = True
    metadata: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        assert isinstance(
            self.kind, TaskProviderReferenceKind
        ), "kind must be a TaskProviderReferenceKind"
        _assert_non_empty_string(self.provider, "provider")
        _assert_non_empty_string(self.reference, "reference")
        if self.owner_scope is not None:
            _assert_non_empty_string(self.owner_scope, "owner_scope")
        if self.expires_at is not None:
            assert isinstance(
                self.expires_at, datetime
            ), "expires_at must be a datetime"
            assert (
                self.expires_at.tzinfo is not None
            ), "expires_at must be timezone-aware"
        if self.mime_type is not None:
            _assert_non_empty_string(self.mime_type, "mime_type")
        if self.size_bucket is not None:
            _assert_non_empty_string(self.size_bucket, "size_bucket")
        if self.identity_hmac is not None:
            _assert_non_empty_string(self.identity_hmac, "identity_hmac")
        assert isinstance(self.durable, bool), "durable must be a bool"
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )

    @property
    def durable_for_queue(self) -> bool:
        return self.durable and self.expires_at is None

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        current = now or datetime.now(UTC)
        assert isinstance(current, datetime)
        assert current.tzinfo is not None, "now must be timezone-aware"
        return self.expires_at <= current

    def execution_metadata(self) -> Mapping[str, object]:
        value: dict[str, object] = {
            "durable": self.durable,
            "kind": self.kind.value,
            "provider": self.provider,
            "reference": self.reference,
        }
        if self.owner_scope is not None:
            value["owner_scope"] = self.owner_scope
        if self.expires_at is not None:
            value["expires_at"] = self.expires_at.isoformat()
        if self.mime_type is not None:
            value["mime_type"] = self.mime_type
        if self.size_bucket is not None:
            value["size_bucket"] = self.size_bucket
        if self.identity_hmac is not None:
            value["identity_hmac"] = self.identity_hmac
        if self.metadata:
            value["metadata"] = self.metadata
        return MappingProxyType(value)

    def summary(self) -> Mapping[str, object]:
        value = dict(self.execution_metadata())
        value.pop("reference", None)
        if "metadata" in value:
            value["metadata"] = _REDACTED_METADATA_SUMMARY
        return MappingProxyType(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskRemoteUrlPolicy:
    enabled: bool = False
    allowed_schemes: tuple[str, ...] = ("https",)
    allow_private_networks: bool = False
    allow_redirects: bool = False
    max_redirects: int = 0
    max_bytes: int | None = None
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        assert isinstance(self.enabled, bool), "enabled must be a bool"
        assert isinstance(
            self.allowed_schemes, tuple
        ), "allowed_schemes must be a tuple"
        for scheme in self.allowed_schemes:
            _assert_non_empty_string(scheme, "allowed_schemes")
            assert (
                scheme == scheme.lower()
            ), "allowed_schemes must be lowercase"
        assert isinstance(
            self.allow_private_networks, bool
        ), "allow_private_networks must be a bool"
        assert isinstance(
            self.allow_redirects, bool
        ), "allow_redirects must be a bool"
        assert isinstance(
            self.max_redirects, int
        ), "max_redirects must be an integer"
        assert not isinstance(
            self.max_redirects, bool
        ), "max_redirects must be an integer"
        assert self.max_redirects >= 0, "max_redirects must be non-negative"
        if self.max_bytes is not None:
            assert isinstance(
                self.max_bytes, int
            ), "max_bytes must be an integer"
            assert not isinstance(
                self.max_bytes, bool
            ), "max_bytes must be an integer"
            assert self.max_bytes > 0, "max_bytes must be positive"
        assert isinstance(
            self.timeout_seconds, int | float
        ), "timeout_seconds must be a number"
        assert not isinstance(
            self.timeout_seconds, bool
        ), "timeout_seconds must be a number"
        assert isfinite(self.timeout_seconds), "timeout_seconds must be finite"
        assert self.timeout_seconds > 0, "timeout_seconds must be positive"

    @classmethod
    def disabled(cls) -> "TaskRemoteUrlPolicy":
        return cls(enabled=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskFileConversionRequest:
    name: str
    options: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.name, "name")
        object.__setattr__(
            self,
            "options",
            freeze_snapshot_metadata(self.options),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskFileDescriptor:
    source_kind: TaskFileSourceKind
    reference: str
    role: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    conversions: tuple[TaskFileConversionRequest, ...] = ()
    provider_reference: TaskProviderReference | None = None
    metadata: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        assert isinstance(
            self.source_kind, TaskFileSourceKind
        ), "source_kind must be a TaskFileSourceKind"
        _assert_non_empty_string(self.reference, "reference")
        if self.role is not None:
            _assert_non_empty_string(self.role, "role")
        if self.mime_type is not None:
            _assert_non_empty_string(self.mime_type, "mime_type")
        if self.size_bytes is not None:
            assert isinstance(
                self.size_bytes, int
            ), "size_bytes must be an integer"
            assert not isinstance(
                self.size_bytes, bool
            ), "size_bytes must be an integer"
            assert self.size_bytes >= 0, "size_bytes must be non-negative"
        if self.sha256 is not None:
            _assert_non_empty_string(self.sha256, "sha256")
        assert isinstance(
            self.conversions, tuple
        ), "conversions must be a tuple"
        for conversion in self.conversions:
            assert isinstance(
                conversion, TaskFileConversionRequest
            ), "conversions must contain TaskFileConversionRequest values"
        if self.provider_reference is not None:
            assert isinstance(
                self.provider_reference,
                TaskProviderReference,
            ), "provider_reference must be a TaskProviderReference"
            assert (
                self.source_kind == TaskFileSourceKind.PROVIDER_REFERENCE
            ), "provider_reference requires provider_reference source kind"
            assert (
                self.provider_reference.reference == self.reference
            ), "provider reference must match descriptor reference"
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )

    @classmethod
    def local_path(
        cls,
        reference: str,
        *,
        role: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        conversions: Iterable[TaskFileConversionRequest] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> "TaskFileDescriptor":
        return cls(
            source_kind=TaskFileSourceKind.LOCAL_PATH,
            reference=reference,
            role=role,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            conversions=tuple(conversions),
            metadata=metadata or {},
        )

    @classmethod
    def provider_reference_descriptor(
        cls,
        reference: str,
        *,
        kind: TaskProviderReferenceKind,
        provider: str,
        owner_scope: str | None = None,
        expires_at: datetime | None = None,
        role: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        size_bucket: str | None = None,
        identity_hmac: str | None = None,
        durable: bool = True,
        metadata: Mapping[str, object] | None = None,
    ) -> "TaskFileDescriptor":
        provider_reference = TaskProviderReference(
            kind=kind,
            provider=provider,
            reference=reference,
            owner_scope=owner_scope,
            expires_at=expires_at,
            mime_type=mime_type,
            size_bucket=size_bucket,
            identity_hmac=identity_hmac,
            durable=durable,
            metadata=metadata or {},
        )
        return cls(
            source_kind=TaskFileSourceKind.PROVIDER_REFERENCE,
            reference=reference,
            role=role,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            provider_reference=provider_reference,
            metadata=metadata or {},
        )

    @classmethod
    def remote_url(
        cls,
        reference: str,
        *,
        role: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        conversions: Iterable[TaskFileConversionRequest] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> "TaskFileDescriptor":
        return cls(
            source_kind=TaskFileSourceKind.REMOTE_URL,
            reference=reference,
            role=role,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            conversions=tuple(conversions),
            metadata=metadata or {},
        )

    @classmethod
    def artifact(
        cls,
        reference: str,
        *,
        role: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        conversions: Iterable[TaskFileConversionRequest] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> "TaskFileDescriptor":
        return cls(
            source_kind=TaskFileSourceKind.ARTIFACT,
            reference=reference,
            role=role,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            conversions=tuple(conversions),
            metadata=metadata or {},
        )

    @classmethod
    def inline_bytes(
        cls,
        reference: str,
        *,
        role: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        conversions: Iterable[TaskFileConversionRequest] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> "TaskFileDescriptor":
        return cls(
            source_kind=TaskFileSourceKind.INLINE_BYTES,
            reference=reference,
            role=role,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            conversions=tuple(conversions),
            metadata=metadata or {},
        )
