from .store import freeze_snapshot_metadata

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


class TaskFileSourceKind(StrEnum):
    LOCAL_PATH = "local_path"
    REMOTE_URL = "remote_url"
    ARTIFACT = "artifact"
    INLINE_BYTES = "inline_bytes"


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskRemoteUrlPolicy:
    enabled: bool = False
    allowed_schemes: tuple[str, ...] = ("https",)
    allow_private_networks: bool = False
    allow_redirects: bool = False
    max_redirects: int = 0
    max_bytes: int | None = None

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


def _assert_non_empty_string(value: str | None, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"
