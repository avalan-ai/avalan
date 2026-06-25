from ..isolation import (
    SandboxEffectiveSettings,
    SandboxEnvironmentPolicy,
)
from ..types import (
    assert_bool as _assert_bool,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ..types import (
    assert_positive_int as _assert_positive_int,
)
from ..types import (
    assert_positive_number as _assert_positive_number,
)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from json import dumps
from posixpath import normpath as normalize_posix_path
from types import MappingProxyType
from typing import TypeVar, cast, final

EnumValue = TypeVar("EnumValue", bound=StrEnum)


class SandboxPlanRequestKind(StrEnum):
    TYPED_TOOL = "typed_tool"
    FLOW_NODE = "flow_node"
    TASK_ATTEMPT = "task_attempt"
    AGENT_SESSION = "agent_session"
    SERVER = "server"
    MCP = "mcp"
    A2A = "a2a"
    RUNTIME_ENVELOPE = "runtime_envelope"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxPlanRequest:
    request_kind: SandboxPlanRequestKind | str
    logical_name: str
    command: str
    argv: Sequence[str]
    cwd: str
    request_id: str | None = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        request_kind = _enum_value(
            self.request_kind,
            SandboxPlanRequestKind,
            "request_kind",
        )
        _assert_non_empty_string(self.logical_name, "logical_name")
        command = _absolute_path(self.command, "command")
        cwd = _absolute_path(self.cwd, "cwd")
        argv = _string_tuple(self.argv, "argv")
        assert argv, "argv must not be empty"
        if self.request_id is not None:
            _assert_non_empty_string(self.request_id, "request_id")
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        if request_kind is SandboxPlanRequestKind.TASK_ATTEMPT:
            assert (
                self.attempt_id is not None
            ), "task attempt requests require attempt_id"
        object.__setattr__(self, "request_kind", request_kind)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "cwd", cwd)

    def to_dict(self) -> dict[str, object]:
        request_kind = cast(SandboxPlanRequestKind, self.request_kind)
        return {
            "request_kind": request_kind.value,
            "logical_name": self.logical_name,
            "command": self.command,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "request_id": self.request_id,
            "attempt_id": self.attempt_id,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxExecutionPlan:
    request: SandboxPlanRequest
    settings: SandboxEffectiveSettings
    environment: Mapping[str, str] = field(default_factory=dict)
    temp_dir: str | None = None
    output_dir: str | None = None
    collect_outputs: bool = False
    cleanup_budget_seconds: float = 2.0
    stream_buffer_bytes: int = 65536

    def __post_init__(self) -> None:
        assert isinstance(self.request, SandboxPlanRequest)
        assert isinstance(self.settings, SandboxEffectiveSettings)
        profile = self.settings.profile
        assert (
            self.request.command in profile.trusted_executables
        ), "sandbox command must be a trusted executable"
        assert _path_allowed_for_read(
            self.request.cwd,
            profile.read_roots,
            profile.write_roots,
            profile.deny_roots,
        ), "cwd must be inside sandbox read/write roots"
        temp_dir = _optional_absolute_path(self.temp_dir, "temp_dir")
        output_dir = _optional_absolute_path(self.output_dir, "output_dir")
        if temp_dir is not None:
            assert _path_inside_any(
                temp_dir,
                profile.scratch_roots,
            ), "temp_dir must be inside sandbox scratch roots"
        if output_dir is not None:
            assert _path_inside_any(
                output_dir,
                profile.output_roots,
            ), "output_dir must be inside sandbox output roots"
        _assert_bool(self.collect_outputs, "collect_outputs")
        _assert_positive_number(
            self.cleanup_budget_seconds,
            "cleanup_budget_seconds",
        )
        _assert_positive_int(self.stream_buffer_bytes, "stream_buffer_bytes")
        environment = _string_mapping(self.environment, "environment")
        _assert_environment_allowed(environment, profile.environment)
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(environment),
        )
        object.__setattr__(self, "temp_dir", temp_dir)
        object.__setattr__(self, "output_dir", output_dir)

    @property
    def plan_fingerprint(self) -> str:
        return _fingerprint(self.canonical_policy_input())

    def canonical_policy_input(self) -> dict[str, object]:
        return {
            "request": self.request.to_dict(),
            "sandbox": self.settings.canonical_policy_input(),
            "environment": dict(sorted(self.environment.items())),
            "temp_dir": self.temp_dir,
            "output_dir": self.output_dir,
            "collect_outputs": self.collect_outputs,
            "cleanup_budget_seconds": self.cleanup_budget_seconds,
            "stream_buffer_bytes": self.stream_buffer_bytes,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "request": self.request.to_dict(),
            "settings": self.settings.to_dict(),
            "environment": dict(self.environment),
            "temp_dir": self.temp_dir,
            "output_dir": self.output_dir,
            "collect_outputs": self.collect_outputs,
            "cleanup_budget_seconds": self.cleanup_budget_seconds,
            "stream_buffer_bytes": self.stream_buffer_bytes,
            "plan_fingerprint": self.plan_fingerprint,
        }


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
        raise AssertionError(
            f"{field_name} contains unsupported value"
        ) from exc


def _absolute_path(value: object, field_name: str) -> str:
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str), f"{field_name} must be a string"
    assert "\x00" not in value, f"{field_name} must not contain NUL"
    assert value.startswith("/"), f"{field_name} must be absolute"
    return normalize_posix_path(value)


def _optional_absolute_path(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _absolute_path(value, field_name)


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    assert isinstance(value, Sequence), f"{field_name} must be a sequence"
    assert not isinstance(
        value, str | bytes
    ), f"{field_name} must be a sequence"
    normalized: list[str] = []
    for item in value:
        _assert_non_empty_string(item, field_name)
        assert isinstance(item, str), f"{field_name} must contain strings"
        assert "\x00" not in item, f"{field_name} must not contain NUL"
        normalized.append(item)
    return tuple(normalized)


def _string_mapping(
    value: Mapping[str, str],
    field_name: str,
) -> dict[str, str]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    normalized: dict[str, str] = {}
    for key, item in value.items():
        _assert_non_empty_string(key, f"{field_name} key")
        _assert_non_empty_string(item, f"{field_name}.{key}")
        normalized[key] = item
    return normalized


def _assert_environment_allowed(
    environment: Mapping[str, str],
    policy: SandboxEnvironmentPolicy,
) -> None:
    variables = policy.variables
    allowlist = policy.allowlist
    allowed_names = set(allowlist) | set(variables)
    for name, value in environment.items():
        assert name in allowed_names, f"environment variable {name} is denied"
        if name in variables:
            assert (
                value == variables[name]
            ), f"environment variable {name} must match sandbox policy"


def _path_allowed_for_read(
    path: str,
    read_roots: Sequence[str],
    write_roots: Sequence[str],
    deny_roots: Sequence[str],
) -> bool:
    if _path_inside_any(path, deny_roots):
        return False
    return _path_inside_any(path, tuple(read_roots) + tuple(write_roots))


def _path_inside_any(path: str, roots: Sequence[str]) -> bool:
    normalized_path = _normalized_parts(path)
    for root in roots:
        normalized_root = _normalized_parts(root)
        if len(normalized_path) >= len(normalized_root) and (
            normalized_path[: len(normalized_root)] == normalized_root
        ):
            return True
    return False


def _normalized_parts(path: str) -> tuple[str, ...]:
    normalized = normalize_posix_path(path)
    return tuple(part for part in normalized.split("/") if part)


def _fingerprint(value: Mapping[str, object]) -> str:
    payload = dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()
