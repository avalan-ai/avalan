from ..types import (
    assert_bool as _assert_bool,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ..types import (
    assert_positive_int as _assert_positive_int,
)
from .conformance import (
    ContainerExecutionScope,
)
from .settings import (
    ContainerAuditEvent,
    ContainerAuditEventType,
    ContainerExecutionResult,
    ContainerOutputPolicy,
    ContainerResultStatus,
)

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from os import O_NOFOLLOW, O_RDONLY, SEEK_SET, fstat, lseek
from os import close as close_fd
from os import open as open_fd
from os import read as read_fd
from pathlib import Path, PurePosixPath
from stat import S_ISDIR, S_ISLNK, S_ISREG
from typing import TypeVar, cast, final

EnumValue = TypeVar("EnumValue", bound=StrEnum)

_DEFAULT_ALLOWED_MEDIA_TYPES = (
    "application/json",
    "text/csv",
    "text/plain",
)
_DEFAULT_DENIED_SIGNATURES = (
    b"MZ",
    b"\x7fELF",
)
_EXTENSION_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".jsonl": "application/json",
    ".log": "text/plain",
    ".md": "text/plain",
    ".ndjson": "application/json",
    ".txt": "text/plain",
}


class ContainerOutputContractType(StrEnum):
    STDOUT = "stdout"
    STDERR = "stderr"
    GENERATED_FILE = "generated_file"
    TASK_ARTIFACT = "task_artifact"
    RUNTIME_ENVELOPE_ARTIFACT = "runtime_envelope_artifact"


class ContainerOutputDecisionType(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    QUARANTINE = "quarantine"


class ContainerOutputDiagnosticCode(StrEnum):
    ABSOLUTE_PATH = "container.output.absolute_path"
    CASE_COLLISION = "container.output.case_collision"
    CONTRACT_DISABLED = "container.output.contract_disabled"
    HARDLINK = "container.output.hardlink"
    RACE_DETECTED = "container.output.race_detected"
    PARTIAL_OUTPUT_DENIED = "container.output.partial_denied"
    PARTIAL_OUTPUT_QUARANTINED = "container.output.partial_quarantined"
    SPECIAL_FILE = "container.output.special_file"
    SYMLINK_ESCAPE = "container.output.symlink_escape"
    TOO_LARGE = "container.output.too_large"
    TOO_MANY_FILES = "container.output.too_many_files"
    TRAVERSAL = "container.output.traversal"
    UNSAFE_MEDIA = "container.output.unsafe_media"
    UNSAFE_OWNERSHIP = "container.output.unsafe_ownership"
    UNSAFE_PERMISSIONS = "container.output.unsafe_permissions"
    UNSAFE_SIGNATURE = "container.output.unsafe_signature"


class ContainerPartialOutputMode(StrEnum):
    DENY = "deny"
    ALLOW = "allow"
    QUARANTINE = "quarantine"


class ContainerPartialOutputReason(StrEnum):
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    RUNTIME_FAILURE = "runtime_failure"


class ContainerArchiveEntryType(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    HARDLINK = "hardlink"
    CHARACTER_DEVICE = "character_device"
    BLOCK_DEVICE = "block_device"
    FIFO = "fifo"
    SOCKET = "socket"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerOutputDiagnostic:
    code: ContainerOutputDiagnosticCode | str
    path: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "code",
            _enum_value(self.code, ContainerOutputDiagnosticCode, "code"),
        )
        _assert_non_empty_string(self.path, "path")
        _assert_non_empty_string(self.message, "message")

    def to_dict(self) -> dict[str, str]:
        code = cast(ContainerOutputDiagnosticCode, self.code)
        return {
            "code": code.value,
            "path": self.path,
            "message": self.message,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerOutputMediaPolicy:
    allowed_media_types: Sequence[str] = _DEFAULT_ALLOWED_MEDIA_TYPES
    denied_signatures: Sequence[bytes] = _DEFAULT_DENIED_SIGNATURES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_media_types",
            tuple(sorted(_string_tuple(self.allowed_media_types, "media"))),
        )
        object.__setattr__(
            self,
            "denied_signatures",
            _bytes_tuple(self.denied_signatures, "denied_signatures"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed_media_types": list(self.allowed_media_types),
            "denied_signatures": [
                signature.hex() for signature in self.denied_signatures
            ],
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerOutputContract:
    contract_type: ContainerOutputContractType | str
    max_bytes: int
    max_files: int = 1
    per_file_bytes: int | None = None
    enabled: bool = True
    media_policy: ContainerOutputMediaPolicy = field(
        default_factory=ContainerOutputMediaPolicy,
    )
    allowed_uids: Sequence[int] | None = None
    allowed_gids: Sequence[int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contract_type",
            _enum_value(
                self.contract_type,
                ContainerOutputContractType,
                "contract_type",
            ),
        )
        _assert_positive_int(self.max_bytes, "max_bytes")
        _assert_positive_int(self.max_files, "max_files")
        if self.per_file_bytes is not None:
            _assert_positive_int(self.per_file_bytes, "per_file_bytes")
        _assert_bool(self.enabled, "enabled")
        assert isinstance(self.media_policy, ContainerOutputMediaPolicy)
        object.__setattr__(
            self,
            "allowed_uids",
            _optional_int_tuple(self.allowed_uids, "allowed_uids"),
        )
        object.__setattr__(
            self,
            "allowed_gids",
            _optional_int_tuple(self.allowed_gids, "allowed_gids"),
        )

    @property
    def file_byte_limit(self) -> int:
        return self.per_file_bytes or self.max_bytes

    @property
    def is_stream(self) -> bool:
        contract_type = cast(ContainerOutputContractType, self.contract_type)
        return contract_type in {
            ContainerOutputContractType.STDOUT,
            ContainerOutputContractType.STDERR,
        }

    def to_dict(self) -> dict[str, object]:
        contract_type = cast(ContainerOutputContractType, self.contract_type)
        return {
            "contract_type": contract_type.value,
            "max_bytes": self.max_bytes,
            "max_files": self.max_files,
            "per_file_bytes": self.per_file_bytes,
            "enabled": self.enabled,
            "media_policy": self.media_policy.to_dict(),
            "allowed_uids": (
                None if self.allowed_uids is None else list(self.allowed_uids)
            ),
            "allowed_gids": (
                None if self.allowed_gids is None else list(self.allowed_gids)
            ),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerPartialOutputPolicy:
    timeout: ContainerPartialOutputMode | str = ContainerPartialOutputMode.DENY
    cancellation: ContainerPartialOutputMode | str = (
        ContainerPartialOutputMode.DENY
    )
    runtime_failure: ContainerPartialOutputMode | str = (
        ContainerPartialOutputMode.QUARANTINE
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timeout",
            _enum_value(self.timeout, ContainerPartialOutputMode, "timeout"),
        )
        object.__setattr__(
            self,
            "cancellation",
            _enum_value(
                self.cancellation,
                ContainerPartialOutputMode,
                "cancellation",
            ),
        )
        object.__setattr__(
            self,
            "runtime_failure",
            _enum_value(
                self.runtime_failure,
                ContainerPartialOutputMode,
                "runtime_failure",
            ),
        )

    def mode_for(
        self,
        reason: ContainerPartialOutputReason | str,
    ) -> ContainerPartialOutputMode:
        resolved = _enum_value(
            reason,
            ContainerPartialOutputReason,
            "reason",
        )
        if resolved is ContainerPartialOutputReason.TIMEOUT:
            return cast(ContainerPartialOutputMode, self.timeout)
        if resolved is ContainerPartialOutputReason.CANCELLATION:
            return cast(ContainerPartialOutputMode, self.cancellation)
        return cast(ContainerPartialOutputMode, self.runtime_failure)

    def to_dict(self) -> dict[str, str]:
        timeout = cast(ContainerPartialOutputMode, self.timeout)
        cancellation = cast(ContainerPartialOutputMode, self.cancellation)
        runtime_failure = cast(
            ContainerPartialOutputMode,
            self.runtime_failure,
        )
        return {
            "timeout": timeout.value,
            "cancellation": cancellation.value,
            "runtime_failure": runtime_failure.value,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerPartialOutput:
    reason: ContainerPartialOutputReason | str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reason",
            _enum_value(
                self.reason,
                ContainerPartialOutputReason,
                "reason",
            ),
        )

    def to_dict(self) -> dict[str, str]:
        reason = cast(ContainerPartialOutputReason, self.reason)
        return {"reason": reason.value}


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerOutputArtifact:
    artifact_type: ContainerOutputContractType | str
    path: str
    size_bytes: int
    media_type: str
    digest: str
    content: bytes | None = None
    quarantined: bool = False

    def __post_init__(self) -> None:
        artifact_type = _enum_value(
            self.artifact_type,
            ContainerOutputContractType,
            "artifact_type",
        )
        assert not _is_stream_type(
            artifact_type
        ), "artifact type must be a file contract"
        _assert_safe_relative_path(self.path)
        assert isinstance(self.size_bytes, int)
        assert self.size_bytes >= 0, "size_bytes must not be negative"
        _assert_non_empty_string(self.media_type, "media_type")
        _assert_digest(self.digest)
        if self.content is not None:
            assert isinstance(self.content, bytes)
            assert len(self.content) == self.size_bytes
        _assert_bool(self.quarantined, "quarantined")
        object.__setattr__(self, "artifact_type", artifact_type)

    def to_dict(self) -> dict[str, object]:
        artifact_type = cast(ContainerOutputContractType, self.artifact_type)
        return {
            "artifact_type": artifact_type.value,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "digest": self.digest,
            "quarantined": self.quarantined,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerArchiveEntry:
    path: str
    entry_type: ContainerArchiveEntryType | str
    size_bytes: int = 0
    mode: int = 0o600
    uid: int | None = None
    gid: int | None = None
    media_type: str | None = None
    signature: bytes = b""

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.path, "path")
        object.__setattr__(
            self,
            "entry_type",
            _enum_value(
                self.entry_type,
                ContainerArchiveEntryType,
                "entry_type",
            ),
        )
        assert isinstance(self.size_bytes, int)
        assert self.size_bytes >= 0, "size_bytes must not be negative"
        _assert_mode(self.mode)
        _assert_optional_non_negative_int(self.uid, "uid")
        _assert_optional_non_negative_int(self.gid, "gid")
        if self.media_type is not None:
            _assert_non_empty_string(self.media_type, "media_type")
        assert isinstance(self.signature, bytes)

    def to_dict(self) -> dict[str, object]:
        entry_type = cast(ContainerArchiveEntryType, self.entry_type)
        return {
            "path": self.path,
            "entry_type": entry_type.value,
            "size_bytes": self.size_bytes,
            "mode": self.mode,
            "uid": self.uid,
            "gid": self.gid,
            "media_type": self.media_type,
            "signature": self.signature.hex(),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerOutputValidationResult:
    decision: ContainerOutputDecisionType | str
    contract: ContainerOutputContract
    artifacts: Sequence[ContainerOutputArtifact] = field(default_factory=tuple)
    diagnostics: Sequence[ContainerOutputDiagnostic] = field(
        default_factory=tuple,
    )
    total_bytes: int = 0
    file_count: int = 0
    partial_output: ContainerPartialOutput | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision",
            _enum_value(
                self.decision,
                ContainerOutputDecisionType,
                "decision",
            ),
        )
        assert isinstance(self.contract, ContainerOutputContract)
        artifacts = tuple(self.artifacts)
        diagnostics = tuple(self.diagnostics)
        for artifact in artifacts:
            assert isinstance(artifact, ContainerOutputArtifact)
        for diagnostic in diagnostics:
            assert isinstance(diagnostic, ContainerOutputDiagnostic)
        assert isinstance(self.total_bytes, int)
        assert self.total_bytes >= 0, "total_bytes must not be negative"
        assert isinstance(self.file_count, int)
        assert self.file_count >= 0, "file_count must not be negative"
        if self.partial_output is not None:
            assert isinstance(self.partial_output, ContainerPartialOutput)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "diagnostics", diagnostics)

    def to_dict(self) -> dict[str, object]:
        decision = cast(ContainerOutputDecisionType, self.decision)
        return {
            "decision": decision.value,
            "contract": self.contract.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
            "total_bytes": self.total_bytes,
            "file_count": self.file_count,
            "partial_output": (
                None
                if self.partial_output is None
                else self.partial_output.to_dict()
            ),
        }

    def to_execution_result(self) -> ContainerExecutionResult:
        decision = cast(ContainerOutputDecisionType, self.decision)
        if decision is ContainerOutputDecisionType.ACCEPT:
            status = ContainerResultStatus.COMPLETED
        elif decision is ContainerOutputDecisionType.REJECT:
            status = ContainerResultStatus.DENIED
        else:
            status = ContainerResultStatus.FAILED
        return ContainerExecutionResult(
            status=status,
            diagnostics=tuple(
                _diagnostic_message(diagnostic)
                for diagnostic in self.diagnostics
            ),
            metadata={
                "output_decision": decision.value,
                "artifact_count": str(len(self.artifacts)),
                "file_count": str(self.file_count),
                "total_bytes": str(self.total_bytes),
            },
        )

    def audit_events(
        self,
        *,
        scope: ContainerExecutionScope | str,
        profile_name: str | None,
        policy_version: str,
    ) -> tuple[ContainerAuditEvent, ...]:
        decision = cast(ContainerOutputDecisionType, self.decision)
        metadata = {
            "decision": decision.value,
            "artifact_count": str(len(self.artifacts)),
            "diagnostic_count": str(len(self.diagnostics)),
        }
        events = [
            ContainerAuditEvent(
                event_type=ContainerAuditEventType.OUTPUT_COPY,
                scope=scope,
                profile_name=profile_name,
                policy_version=policy_version,
                metadata=metadata,
            )
        ]
        if decision is ContainerOutputDecisionType.REJECT:
            events.append(
                ContainerAuditEvent(
                    event_type=ContainerAuditEventType.DENIAL,
                    scope=scope,
                    profile_name=profile_name,
                    policy_version=policy_version,
                    metadata=metadata,
                )
            )
        if decision is ContainerOutputDecisionType.QUARANTINE:
            events.append(
                ContainerAuditEvent(
                    event_type=ContainerAuditEventType.FAILURE,
                    scope=scope,
                    profile_name=profile_name,
                    policy_version=policy_version,
                    metadata=metadata,
                )
            )
        events.append(
            ContainerAuditEvent(
                event_type=ContainerAuditEventType.RESULT_RECORDED,
                scope=scope,
                profile_name=profile_name,
                policy_version=policy_version,
                metadata=metadata,
            )
        )
        return tuple(events)


def output_contracts_from_policy(
    policy: ContainerOutputPolicy,
) -> tuple[ContainerOutputContract, ...]:
    assert isinstance(policy, ContainerOutputPolicy)
    artifact_bytes = max(policy.max_artifact_bytes, 1)
    return (
        ContainerOutputContract(
            contract_type=ContainerOutputContractType.STDOUT,
            max_bytes=policy.max_stdout_bytes,
        ),
        ContainerOutputContract(
            contract_type=ContainerOutputContractType.STDERR,
            max_bytes=policy.max_stderr_bytes,
        ),
        ContainerOutputContract(
            contract_type=ContainerOutputContractType.GENERATED_FILE,
            max_bytes=artifact_bytes,
            max_files=100,
            enabled=policy.allow_artifacts,
        ),
        ContainerOutputContract(
            contract_type=ContainerOutputContractType.TASK_ARTIFACT,
            max_bytes=artifact_bytes,
            max_files=100,
            enabled=policy.allow_artifacts,
        ),
        ContainerOutputContract(
            contract_type=(
                ContainerOutputContractType.RUNTIME_ENVELOPE_ARTIFACT
            ),
            max_bytes=artifact_bytes,
            max_files=100,
            enabled=policy.allow_artifacts,
        ),
    )


def validate_output_stream(
    content: bytes,
    contract: ContainerOutputContract,
    *,
    partial_output: ContainerPartialOutput | None = None,
    partial_policy: ContainerPartialOutputPolicy | None = None,
) -> ContainerOutputValidationResult:
    assert isinstance(content, bytes)
    assert isinstance(contract, ContainerOutputContract)
    resolved_policy = partial_policy or ContainerPartialOutputPolicy()
    assert isinstance(resolved_policy, ContainerPartialOutputPolicy)
    assert contract.is_stream, "stream validation requires a stream contract"
    diagnostics = _partial_diagnostics(partial_output, resolved_policy)
    if _partial_mode(partial_output, resolved_policy) is (
        ContainerPartialOutputMode.DENY
    ):
        return _result(
            contract,
            diagnostics,
            len(content),
            0,
            partial_output=partial_output,
        )
    if not contract.enabled:
        diagnostics.append(
            _diagnostic(
                ContainerOutputDiagnosticCode.CONTRACT_DISABLED,
                "<stream>",
                "output stream collection is disabled",
            )
        )
    if len(content) > contract.max_bytes:
        diagnostics.append(
            _diagnostic(
                ContainerOutputDiagnosticCode.TOO_LARGE,
                "<stream>",
                "output stream exceeds byte limit",
            )
        )
    return _result(
        contract,
        diagnostics,
        len(content),
        0,
        partial_output=partial_output,
    )


def validate_copied_outputs(
    root: str,
    contract: ContainerOutputContract,
    *,
    partial_output: ContainerPartialOutput | None = None,
    partial_policy: ContainerPartialOutputPolicy | None = None,
) -> ContainerOutputValidationResult:
    _assert_non_empty_string(root, "root")
    assert isinstance(contract, ContainerOutputContract)
    resolved_policy = partial_policy or ContainerPartialOutputPolicy()
    assert isinstance(resolved_policy, ContainerPartialOutputPolicy)
    assert not contract.is_stream, "copied output validation needs artifacts"
    diagnostics = _partial_diagnostics(partial_output, resolved_policy)
    if not contract.enabled:
        diagnostics.append(
            _diagnostic(
                ContainerOutputDiagnosticCode.CONTRACT_DISABLED,
                ".",
                "artifact collection is disabled",
            )
        )
        return _result(
            contract,
            diagnostics,
            0,
            0,
            partial_output=partial_output,
        )
    if _partial_mode(partial_output, resolved_policy) is (
        ContainerPartialOutputMode.DENY
    ):
        return _result(
            contract,
            diagnostics,
            0,
            0,
            partial_output=partial_output,
        )
    root_path = Path(root)
    assert root_path.exists() and root_path.is_dir(), "output root must exist"
    artifacts: list[ContainerOutputArtifact] = []
    total_bytes = 0
    file_count = 0
    for path in sorted(root_path.rglob("*"), key=lambda item: item.as_posix()):
        relative_path = path.relative_to(root_path).as_posix()
        stat_result = path.lstat()
        mode = stat_result.st_mode
        entry_diagnostics = _relative_path_diagnostics(relative_path)
        entry_diagnostics.extend(
            _mode_and_owner_diagnostics(
                relative_path,
                mode,
                stat_result.st_uid,
                stat_result.st_gid,
                contract,
            )
        )
        if S_ISLNK(mode):
            entry_diagnostics.append(
                _diagnostic(
                    ContainerOutputDiagnosticCode.SYMLINK_ESCAPE,
                    relative_path,
                    "symlink outputs are denied",
                )
            )
            diagnostics.extend(entry_diagnostics)
            continue
        if S_ISDIR(mode):
            diagnostics.extend(entry_diagnostics)
            continue
        if not S_ISREG(mode):
            entry_diagnostics.append(
                _diagnostic(
                    ContainerOutputDiagnosticCode.SPECIAL_FILE,
                    relative_path,
                    "special output files are denied",
                )
            )
            diagnostics.extend(entry_diagnostics)
            continue
        file_count += 1
        entry_diagnostics.extend(
            _file_limit_diagnostics(
                relative_path,
                file_count,
                stat_result.st_size,
                total_bytes + stat_result.st_size,
                contract,
            )
        )
        if entry_diagnostics:
            diagnostics.extend(entry_diagnostics)
            total_bytes += stat_result.st_size
            continue
        opened_file = _read_validated_file(
            path,
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_mode,
            stat_result.st_size,
            stat_result.st_uid,
            stat_result.st_gid,
        )
        if opened_file is None:
            entry_diagnostics.append(
                _diagnostic(
                    ContainerOutputDiagnosticCode.RACE_DETECTED,
                    relative_path,
                    "output changed while being collected",
                )
            )
            diagnostics.extend(entry_diagnostics)
            continue
        signature, digest = opened_file
        media_type = _media_type_for_path(relative_path, None)
        entry_diagnostics.extend(
            _media_diagnostics(
                relative_path,
                media_type,
                signature,
                contract.media_policy,
            )
        )
        diagnostics.extend(entry_diagnostics)
        if not entry_diagnostics:
            artifacts.append(
                ContainerOutputArtifact(
                    artifact_type=contract.contract_type,
                    path=relative_path,
                    size_bytes=stat_result.st_size,
                    media_type=media_type,
                    digest=digest,
                    quarantined=_partial_mode(
                        partial_output,
                        resolved_policy,
                    )
                    is ContainerPartialOutputMode.QUARANTINE,
                )
            )
        total_bytes += stat_result.st_size
    return _result(
        contract,
        diagnostics,
        total_bytes,
        file_count,
        artifacts=artifacts,
        partial_output=partial_output,
    )


def validate_archive_entries(
    entries: Sequence[ContainerArchiveEntry],
    contract: ContainerOutputContract,
    *,
    partial_output: ContainerPartialOutput | None = None,
    partial_policy: ContainerPartialOutputPolicy | None = None,
) -> ContainerOutputValidationResult:
    assert isinstance(contract, ContainerOutputContract)
    resolved_policy = partial_policy or ContainerPartialOutputPolicy()
    assert isinstance(resolved_policy, ContainerPartialOutputPolicy)
    assert not contract.is_stream, "archive validation needs artifacts"
    diagnostics = _partial_diagnostics(partial_output, resolved_policy)
    if not contract.enabled:
        diagnostics.append(
            _diagnostic(
                ContainerOutputDiagnosticCode.CONTRACT_DISABLED,
                ".",
                "artifact collection is disabled",
            )
        )
        return _result(
            contract,
            diagnostics,
            0,
            0,
            partial_output=partial_output,
        )
    if _partial_mode(partial_output, resolved_policy) is (
        ContainerPartialOutputMode.DENY
    ):
        return _result(
            contract,
            diagnostics,
            0,
            0,
            partial_output=partial_output,
        )
    artifacts: list[ContainerOutputArtifact] = []
    total_bytes = 0
    file_count = 0
    entries_tuple = tuple(entries)
    collision_counts: dict[str, int] = {}
    for entry in entries_tuple:
        assert isinstance(entry, ContainerArchiveEntry)
        collision_key = _canonical_relative_path(entry.path).casefold()
        collision_counts[collision_key] = (
            collision_counts.get(collision_key, 0) + 1
        )
    colliding_keys = {
        key for key, count in collision_counts.items() if count > 1
    }
    for entry in entries_tuple:
        assert isinstance(entry, ContainerArchiveEntry)
        entry_type = cast(ContainerArchiveEntryType, entry.entry_type)
        canonical_path = _canonical_relative_path(entry.path)
        collision_key = canonical_path.casefold()
        entry_diagnostics = _archive_path_diagnostics(entry.path)
        if collision_key in colliding_keys:
            entry_diagnostics.append(
                _diagnostic(
                    ContainerOutputDiagnosticCode.CASE_COLLISION,
                    entry.path,
                    "archive path collides by case",
                )
            )
        entry_diagnostics.extend(
            _mode_and_owner_diagnostics(
                canonical_path,
                entry.mode,
                entry.uid,
                entry.gid,
                contract,
            )
        )
        if entry_type is ContainerArchiveEntryType.SYMLINK:
            entry_diagnostics.append(
                _diagnostic(
                    ContainerOutputDiagnosticCode.SYMLINK_ESCAPE,
                    entry.path,
                    "archive symlinks are denied",
                )
            )
        if entry_type is ContainerArchiveEntryType.HARDLINK:
            entry_diagnostics.append(
                _diagnostic(
                    ContainerOutputDiagnosticCode.HARDLINK,
                    entry.path,
                    "archive hardlinks are denied",
                )
            )
        if entry_type in {
            ContainerArchiveEntryType.BLOCK_DEVICE,
            ContainerArchiveEntryType.CHARACTER_DEVICE,
            ContainerArchiveEntryType.FIFO,
            ContainerArchiveEntryType.SOCKET,
        }:
            entry_diagnostics.append(
                _diagnostic(
                    ContainerOutputDiagnosticCode.SPECIAL_FILE,
                    entry.path,
                    "archive special files are denied",
                )
            )
        if entry_type is ContainerArchiveEntryType.FILE:
            file_count += 1
            entry_diagnostics.extend(
                _file_limit_diagnostics(
                    canonical_path,
                    file_count,
                    entry.size_bytes,
                    total_bytes + entry.size_bytes,
                    contract,
                )
            )
            media_type = _media_type_for_path(
                canonical_path,
                entry.media_type,
            )
            entry_diagnostics.extend(
                _media_diagnostics(
                    canonical_path,
                    media_type,
                    entry.signature,
                    contract.media_policy,
                )
            )
            total_bytes += entry.size_bytes
            if not entry_diagnostics:
                artifacts.append(
                    ContainerOutputArtifact(
                        artifact_type=contract.contract_type,
                        path=canonical_path,
                        size_bytes=entry.size_bytes,
                        media_type=media_type,
                        digest=_metadata_digest(entry),
                        quarantined=_partial_mode(
                            partial_output,
                            resolved_policy,
                        )
                        is ContainerPartialOutputMode.QUARANTINE,
                    )
                )
        diagnostics.extend(entry_diagnostics)
    return _result(
        contract,
        diagnostics,
        total_bytes,
        file_count,
        artifacts=artifacts,
        partial_output=partial_output,
    )


def _result(
    contract: ContainerOutputContract,
    diagnostics: Sequence[ContainerOutputDiagnostic],
    total_bytes: int,
    file_count: int,
    *,
    artifacts: Sequence[ContainerOutputArtifact] = (),
    partial_output: ContainerPartialOutput | None = None,
) -> ContainerOutputValidationResult:
    decision = ContainerOutputDecisionType.ACCEPT
    diagnostic_codes = {
        cast(ContainerOutputDiagnosticCode, diagnostic.code)
        for diagnostic in diagnostics
    }
    if diagnostic_codes - {
        ContainerOutputDiagnosticCode.PARTIAL_OUTPUT_QUARANTINED,
    }:
        decision = ContainerOutputDecisionType.REJECT
    elif (
        ContainerOutputDiagnosticCode.PARTIAL_OUTPUT_QUARANTINED
        in diagnostic_codes
    ):
        decision = ContainerOutputDecisionType.QUARANTINE
    return ContainerOutputValidationResult(
        decision=decision,
        contract=contract,
        artifacts=artifacts,
        diagnostics=diagnostics,
        total_bytes=total_bytes,
        file_count=file_count,
        partial_output=partial_output,
    )


def _partial_mode(
    partial_output: ContainerPartialOutput | None,
    policy: ContainerPartialOutputPolicy,
) -> ContainerPartialOutputMode | None:
    if partial_output is None:
        return None
    return policy.mode_for(partial_output.reason)


def _partial_diagnostics(
    partial_output: ContainerPartialOutput | None,
    policy: ContainerPartialOutputPolicy,
) -> list[ContainerOutputDiagnostic]:
    mode = _partial_mode(partial_output, policy)
    if partial_output is None or mode is ContainerPartialOutputMode.ALLOW:
        return []
    reason = cast(ContainerPartialOutputReason, partial_output.reason)
    if mode is ContainerPartialOutputMode.DENY:
        return [
            _diagnostic(
                ContainerOutputDiagnosticCode.PARTIAL_OUTPUT_DENIED,
                "<partial>",
                f"partial output after {reason.value} is denied",
            )
        ]
    return [
        _diagnostic(
            ContainerOutputDiagnosticCode.PARTIAL_OUTPUT_QUARANTINED,
            "<partial>",
            f"partial output after {reason.value} is quarantined",
        )
    ]


def _relative_path_diagnostics(
    path: str,
) -> list[ContainerOutputDiagnostic]:
    diagnostics: list[ContainerOutputDiagnostic] = []
    try:
        _assert_safe_relative_path(path)
    except AssertionError:
        diagnostics.append(
            _diagnostic(
                ContainerOutputDiagnosticCode.TRAVERSAL,
                path,
                "output path is not a safe relative path",
            )
        )
    return diagnostics


def _archive_path_diagnostics(path: str) -> list[ContainerOutputDiagnostic]:
    if path.startswith("/") or path.startswith("//"):
        return [
            _diagnostic(
                ContainerOutputDiagnosticCode.ABSOLUTE_PATH,
                path,
                "archive entry uses an absolute path",
            )
        ]
    return _relative_path_diagnostics(_canonical_relative_path(path))


def _mode_and_owner_diagnostics(
    path: str,
    mode: int,
    uid: int | None,
    gid: int | None,
    contract: ContainerOutputContract,
) -> list[ContainerOutputDiagnostic]:
    diagnostics: list[ContainerOutputDiagnostic] = []
    if _unsafe_mode(mode):
        diagnostics.append(
            _diagnostic(
                ContainerOutputDiagnosticCode.UNSAFE_PERMISSIONS,
                path,
                "output permissions are unsafe",
            )
        )
    if contract.allowed_uids is not None and uid not in contract.allowed_uids:
        diagnostics.append(
            _diagnostic(
                ContainerOutputDiagnosticCode.UNSAFE_OWNERSHIP,
                path,
                "output uid is not allowed",
            )
        )
    if contract.allowed_gids is not None and gid not in contract.allowed_gids:
        diagnostics.append(
            _diagnostic(
                ContainerOutputDiagnosticCode.UNSAFE_OWNERSHIP,
                path,
                "output gid is not allowed",
            )
        )
    return diagnostics


def _file_limit_diagnostics(
    path: str,
    file_count: int,
    size_bytes: int,
    total_bytes: int,
    contract: ContainerOutputContract,
) -> list[ContainerOutputDiagnostic]:
    diagnostics: list[ContainerOutputDiagnostic] = []
    if file_count > contract.max_files:
        diagnostics.append(
            _diagnostic(
                ContainerOutputDiagnosticCode.TOO_MANY_FILES,
                path,
                "output file count exceeds limit",
            )
        )
    if size_bytes > contract.file_byte_limit:
        diagnostics.append(
            _diagnostic(
                ContainerOutputDiagnosticCode.TOO_LARGE,
                path,
                "output file exceeds per-file byte limit",
            )
        )
    if total_bytes > contract.max_bytes:
        diagnostics.append(
            _diagnostic(
                ContainerOutputDiagnosticCode.TOO_LARGE,
                path,
                "output files exceed total byte limit",
            )
        )
    return diagnostics


def _media_diagnostics(
    path: str,
    media_type: str,
    signature: bytes,
    policy: ContainerOutputMediaPolicy,
) -> list[ContainerOutputDiagnostic]:
    diagnostics: list[ContainerOutputDiagnostic] = []
    if media_type not in policy.allowed_media_types:
        diagnostics.append(
            _diagnostic(
                ContainerOutputDiagnosticCode.UNSAFE_MEDIA,
                path,
                "output media type is not allowed",
            )
        )
    for denied in policy.denied_signatures:
        if signature.startswith(denied):
            diagnostics.append(
                _diagnostic(
                    ContainerOutputDiagnosticCode.UNSAFE_SIGNATURE,
                    path,
                    "output file signature is denied",
                )
            )
    return diagnostics


def _diagnostic(
    code: ContainerOutputDiagnosticCode,
    path: str,
    message: str,
) -> ContainerOutputDiagnostic:
    return ContainerOutputDiagnostic(code=code, path=path, message=message)


def _diagnostic_message(diagnostic: ContainerOutputDiagnostic) -> str:
    code = cast(ContainerOutputDiagnosticCode, diagnostic.code)
    return f"{code.value}:{diagnostic.path}:{diagnostic.message}"


def _media_type_for_path(
    path: str,
    declared_media_type: str | None,
) -> str:
    if declared_media_type is not None:
        return declared_media_type
    suffix = PurePosixPath(path).suffix.lower()
    return _EXTENSION_MEDIA_TYPES.get(suffix, "application/octet-stream")


def _read_validated_file(
    path: Path,
    expected_device: int,
    expected_inode: int,
    expected_mode: int,
    expected_size: int,
    expected_uid: int,
    expected_gid: int,
) -> tuple[bytes, str] | None:
    try:
        file_descriptor = open_fd(str(path), O_RDONLY | O_NOFOLLOW)
    except OSError:
        return None
    try:
        try:
            stat_result = fstat(file_descriptor)
            if not _same_validated_file(
                stat_result.st_dev,
                stat_result.st_ino,
                stat_result.st_mode,
                stat_result.st_size,
                stat_result.st_uid,
                stat_result.st_gid,
                expected_device,
                expected_inode,
                expected_mode,
                expected_size,
                expected_uid,
                expected_gid,
            ):
                return None
            signature, digest = _read_fd_signature_and_digest(
                file_descriptor,
                expected_size,
            )
            after_stat = fstat(file_descriptor)
            if not _same_validated_file(
                after_stat.st_dev,
                after_stat.st_ino,
                after_stat.st_mode,
                after_stat.st_size,
                after_stat.st_uid,
                after_stat.st_gid,
                expected_device,
                expected_inode,
                expected_mode,
                expected_size,
                expected_uid,
                expected_gid,
            ):
                return None
        except OSError:
            return None
        return signature, digest
    finally:
        close_fd(file_descriptor)


def _same_validated_file(
    actual_device: int,
    actual_inode: int,
    actual_mode: int,
    actual_size: int,
    actual_uid: int,
    actual_gid: int,
    expected_device: int,
    expected_inode: int,
    expected_mode: int,
    expected_size: int,
    expected_uid: int,
    expected_gid: int,
) -> bool:
    return (
        actual_device == expected_device
        and actual_inode == expected_inode
        and S_ISREG(actual_mode)
        and S_ISREG(expected_mode)
        and actual_size == expected_size
        and actual_uid == expected_uid
        and actual_gid == expected_gid
        and (actual_mode & 0o7777) == (expected_mode & 0o7777)
    )


def _read_fd_signature_and_digest(
    file_descriptor: int,
    expected_size: int,
) -> tuple[bytes, str]:
    digest = sha256()
    signature = b""
    remaining_size = expected_size
    lseek(file_descriptor, 0, SEEK_SET)
    while remaining_size > 0:
        chunk = read_fd(file_descriptor, min(65536, remaining_size))
        if not chunk:
            break
        remaining_size -= len(chunk)
        if len(signature) < 16:
            remaining = 16 - len(signature)
            signature += chunk[:remaining]
        digest.update(chunk)
    if remaining_size != 0 or read_fd(file_descriptor, 1):
        raise OSError("output changed while being read")
    return signature, f"sha256:{digest.hexdigest()}"


def _metadata_digest(entry: ContainerArchiveEntry) -> str:
    digest = sha256()
    digest.update(entry.path.encode("utf-8"))
    digest.update(str(entry.size_bytes).encode("ascii"))
    digest.update(entry.signature)
    return f"sha256:{digest.hexdigest()}"


def _assert_safe_relative_path(path: str) -> None:
    _assert_non_empty_string(path, "path")
    assert "\x00" not in path, "path must not contain NUL"
    assert not path.startswith("/"), "path must be relative"
    assert not path.startswith("//"), "path must be relative"
    assert "\\" not in path, "path must use POSIX separators"
    parts = PurePosixPath(path).parts
    assert parts, "path must not be empty"
    assert not any(
        part in {"", ".", ".."} for part in parts
    ), "path must not traverse"


def _canonical_relative_path(path: str) -> str:
    _assert_non_empty_string(path, "path")
    return PurePosixPath(path).as_posix()


def _assert_digest(value: str) -> None:
    _assert_non_empty_string(value, "digest")
    assert value.startswith("sha256:"), "digest must be sha256"
    hex_value = value.removeprefix("sha256:")
    assert len(hex_value) == 64, "digest must be sha256 hex"
    int(hex_value, 16)


def _assert_mode(mode: int) -> None:
    assert isinstance(mode, int)
    assert 0 <= mode <= 0o7777, "mode must be a permission mode"


def _assert_optional_non_negative_int(
    value: int | None,
    field_name: str,
) -> None:
    if value is None:
        return
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert value >= 0, f"{field_name} must not be negative"


def _unsafe_mode(mode: int) -> bool:
    permissions = mode & 0o7777
    return bool(permissions & 0o7000 or permissions & 0o022)


def _is_stream_type(contract_type: ContainerOutputContractType) -> bool:
    return contract_type in {
        ContainerOutputContractType.STDOUT,
        ContainerOutputContractType.STDERR,
    }


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    assert isinstance(value, Sequence) and not isinstance(
        value, str
    ), f"{field_name} must be a sequence"
    result = tuple(value)
    for item in result:
        _assert_non_empty_string(item, field_name)
        assert isinstance(item, str)
    return result


def _bytes_tuple(value: object, field_name: str) -> tuple[bytes, ...]:
    assert isinstance(value, Sequence) and not isinstance(
        value, bytes
    ), f"{field_name} must be a sequence"
    result = tuple(value)
    for item in result:
        assert isinstance(item, bytes), f"{field_name} must contain bytes"
        assert item, f"{field_name} must not contain empty bytes"
    return result


def _optional_int_tuple(
    value: Sequence[int] | None,
    field_name: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    assert isinstance(value, Sequence), f"{field_name} must be a sequence"
    result = tuple(value)
    for item in result:
        assert isinstance(item, int), f"{field_name} must contain integers"
        assert item >= 0, f"{field_name} must not contain negative integers"
    return result


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
