from ..types import (
    assert_bool as _assert_bool,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import final


class ContainerBackend(StrEnum):
    NONE = "none"
    DOCKER = "docker"
    PODMAN = "podman"
    NERDCTL = "nerdctl"
    APPLE_CONTAINER = "apple-container"
    WINDOWS_DOCKER = "windows-docker"
    AUTO = "auto"


class ContainerConformanceSlice(StrEnum):
    FIRST_RELEASE = "first_release"
    LATER = "later"


class ContainerDiagnosticCategory(StrEnum):
    UNSUPPORTED = "unsupported"
    VALUE = "value"
    RUNTIME = "runtime"


class ContainerDiagnosticCode(StrEnum):
    BACKEND_REQUIRED = "container.backend_required"
    BACKEND_UNAVAILABLE = "container.backend_unavailable"
    UNSUPPORTED_SYNTAX = "container.unsupported_syntax"


class ContainerExecutionScope(StrEnum):
    CORE_CONTRACT = "core_contract"
    SHELL_CONTAINER_EXECUTION = "shell_container_execution"
    DURABLE_WORKFLOW = "durable_workflow"
    BACKEND_BREADTH = "backend_breadth"
    RUNTIME_ENVELOPE = "runtime_envelope"
    MODEL_BACKEND = "model_backend"
    ADVANCED_RUNTIME_FEATURES = "advanced_runtime_features"


class ContainerSurface(StrEnum):
    SDK = "sdk"
    CLI = "cli"
    AGENT_TOML = "agent_toml"
    FLOW_TOML = "flow_toml"
    TASK_TOML = "task_toml"
    SERVER = "server"
    MCP = "mcp"
    A2A = "a2a"
    RUNTIME_ENVELOPE = "runtime_envelope"
    MODEL_BACKEND = "model_backend"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerDiagnostic:
    code: ContainerDiagnosticCode
    path: str
    message: str
    hint: str
    category: ContainerDiagnosticCategory

    def __post_init__(self) -> None:
        assert isinstance(
            self.code, ContainerDiagnosticCode
        ), "code must be a container diagnostic code"
        _assert_non_empty_string(self.path, "path")
        _assert_non_empty_string(self.message, "message")
        _assert_non_empty_string(self.hint, "hint")
        assert isinstance(
            self.category, ContainerDiagnosticCategory
        ), "category must be a container diagnostic category"

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "path": self.path,
            "category": self.category.value,
            "message": self.message,
            "hint": self.hint,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerConformanceTarget:
    scope: ContainerExecutionScope
    release_slice: ContainerConformanceSlice
    description: str

    def __post_init__(self) -> None:
        assert isinstance(
            self.scope, ContainerExecutionScope
        ), "scope must be a container execution scope"
        assert isinstance(
            self.release_slice, ContainerConformanceSlice
        ), "release_slice must be a container conformance slice"
        _assert_non_empty_string(self.description, "description")


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerConformancePlan:
    objective: str
    targets: tuple[ContainerConformanceTarget, ...]
    default_ci_jobs: tuple[str, ...]
    optional_runtime_ci_jobs: tuple[str, ...]
    unsupported_surface_paths: Mapping[ContainerSurface, tuple[str, ...]]
    promoted_integration_backends: tuple[ContainerBackend, ...] = ()
    optional_integration_backends: tuple[ContainerBackend, ...] = ()
    opt_in_integration_backends: tuple[ContainerBackend, ...] = ()
    backward_compatibility_required: bool = False
    reject_unknown_container_sections: bool = True

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.objective, "objective")
        assert self.targets, "targets must not be empty"
        for target in self.targets:
            assert isinstance(
                target, ContainerConformanceTarget
            ), "targets must contain container conformance targets"
        _assert_non_empty_string_tuple(
            self.default_ci_jobs,
            "default_ci_jobs",
        )
        _assert_non_empty_string_tuple(
            self.optional_runtime_ci_jobs,
            "optional_runtime_ci_jobs",
        )
        assert isinstance(
            self.unsupported_surface_paths, Mapping
        ), "unsupported_surface_paths must be a mapping"
        for field_name in (
            "promoted_integration_backends",
            "optional_integration_backends",
            "opt_in_integration_backends",
        ):
            for backend in getattr(self, field_name):
                assert isinstance(
                    backend,
                    ContainerBackend,
                ), f"{field_name} must contain container backends"
        frozen_paths: dict[ContainerSurface, tuple[str, ...]] = {}
        for surface, paths in self.unsupported_surface_paths.items():
            assert isinstance(
                surface, ContainerSurface
            ), "unsupported_surface_paths keys must be container surfaces"
            _assert_non_empty_string_tuple(
                paths,
                "unsupported_surface_paths",
            )
            frozen_paths[surface] = tuple(paths)
        _assert_bool(
            self.backward_compatibility_required,
            "backward_compatibility_required",
        )
        _assert_bool(
            self.reject_unknown_container_sections,
            "reject_unknown_container_sections",
        )
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(
            self,
            "default_ci_jobs",
            tuple(self.default_ci_jobs),
        )
        object.__setattr__(
            self,
            "optional_runtime_ci_jobs",
            tuple(self.optional_runtime_ci_jobs),
        )
        object.__setattr__(
            self,
            "unsupported_surface_paths",
            MappingProxyType(frozen_paths),
        )
        object.__setattr__(
            self,
            "promoted_integration_backends",
            tuple(self.promoted_integration_backends),
        )
        object.__setattr__(
            self,
            "optional_integration_backends",
            tuple(self.optional_integration_backends),
        )
        object.__setattr__(
            self,
            "opt_in_integration_backends",
            tuple(self.opt_in_integration_backends),
        )

    @property
    def first_release_scopes(self) -> tuple[ContainerExecutionScope, ...]:
        return tuple(
            target.scope
            for target in self.targets
            if target.release_slice is ContainerConformanceSlice.FIRST_RELEASE
        )

    @property
    def later_scopes(self) -> tuple[ContainerExecutionScope, ...]:
        return tuple(
            target.scope
            for target in self.targets
            if target.release_slice is ContainerConformanceSlice.LATER
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerExecutionSettings:
    backend: ContainerBackend | str = ContainerBackend.NONE
    required: bool = False
    scope: ContainerExecutionScope | str = (
        ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
    )
    profile: str | None = None

    def __post_init__(self) -> None:
        backend = _container_backend(self.backend)
        scope = _container_scope(self.scope)
        _assert_bool(self.required, "required")
        if self.profile is not None:
            _assert_non_empty_string(self.profile, "profile")
        assert (
            backend is not ContainerBackend.NONE or self.profile is None
        ), "profile requires a container backend"
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "scope", scope)

    @property
    def enabled(self) -> bool:
        return self.backend is not ContainerBackend.NONE


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendResolution:
    backend: ContainerBackend | None
    direct_execution_allowed: bool
    diagnostics: tuple[ContainerDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.backend is not None:
            assert isinstance(
                self.backend, ContainerBackend
            ), "backend must be a container backend"
        _assert_bool(
            self.direct_execution_allowed,
            "direct_execution_allowed",
        )
        for diagnostic in self.diagnostics:
            assert isinstance(
                diagnostic, ContainerDiagnostic
            ), "diagnostics must contain container diagnostics"
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def ok(self) -> bool:
        return not self.diagnostics


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class _ContainerSyntaxRule:
    path: str
    expected_value: object = None
    match_any_value: bool = True

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.path, "path")
        _assert_bool(self.match_any_value, "match_any_value")


def _assert_non_empty_string_tuple(
    value: object,
    field_name: str,
) -> None:
    assert isinstance(value, tuple), f"{field_name} must be a tuple"
    assert value, f"{field_name} must not be empty"
    for item in value:
        _assert_non_empty_string(item, field_name)


CONFORMANCE_PLAN = ContainerConformancePlan(
    objective="Implement full specs/CONTAINERS.md coverage iteratively.",
    targets=(
        ContainerConformanceTarget(
            scope=ContainerExecutionScope.CORE_CONTRACT,
            release_slice=ContainerConformanceSlice.FIRST_RELEASE,
            description="Shared settings, diagnostics, and conformance.",
        ),
        ContainerConformanceTarget(
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
            release_slice=ContainerConformanceSlice.FIRST_RELEASE,
            description="Shell command container execution.",
        ),
        ContainerConformanceTarget(
            scope=ContainerExecutionScope.DURABLE_WORKFLOW,
            release_slice=ContainerConformanceSlice.LATER,
            description="Flow, task, worker, and durable review support.",
        ),
        ContainerConformanceTarget(
            scope=ContainerExecutionScope.BACKEND_BREADTH,
            release_slice=ContainerConformanceSlice.LATER,
            description="Docker, Podman, nerdctl, Apple, and Windows breadth.",
        ),
        ContainerConformanceTarget(
            scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
            release_slice=ContainerConformanceSlice.LATER,
            description="Whole-runtime container boundaries.",
        ),
        ContainerConformanceTarget(
            scope=ContainerExecutionScope.MODEL_BACKEND,
            release_slice=ContainerConformanceSlice.LATER,
            description="Eligible model backend container execution.",
        ),
        ContainerConformanceTarget(
            scope=ContainerExecutionScope.ADVANCED_RUNTIME_FEATURES,
            release_slice=ContainerConformanceSlice.LATER,
            description="Pooling, builds, provenance, and service profiles.",
        ),
    ),
    default_ci_jobs=(
        "container conformance unit tests",
        "container fake-e2e no-fallback tests",
    ),
    optional_runtime_ci_jobs=(
        "docker runtime e2e",
        "podman runtime e2e",
        "nerdctl runtime e2e",
        "apple-container runtime e2e",
        "windows-docker runtime e2e",
    ),
    promoted_integration_backends=(ContainerBackend.DOCKER,),
    optional_integration_backends=(
        ContainerBackend.PODMAN,
        ContainerBackend.NERDCTL,
        ContainerBackend.WINDOWS_DOCKER,
    ),
    opt_in_integration_backends=(ContainerBackend.APPLE_CONTAINER,),
    unsupported_surface_paths=MappingProxyType(
        {
            ContainerSurface.TASK_TOML: (
                "container",
                "execution.container",
            ),
            ContainerSurface.SERVER: ("request.container",),
            ContainerSurface.MCP: ("request.container",),
            ContainerSurface.A2A: ("request.container",),
            ContainerSurface.RUNTIME_ENVELOPE: ("runtime.container",),
            ContainerSurface.MODEL_BACKEND: ("model.container",),
        }
    ),
)

_CONTAINER_SYNTAX_RULES: Mapping[
    ContainerSurface, tuple[_ContainerSyntaxRule, ...]
] = MappingProxyType(
    {
        ContainerSurface.TASK_TOML: (
            _ContainerSyntaxRule(path="container"),
            _ContainerSyntaxRule(path="execution.container"),
        ),
        ContainerSurface.SERVER: (
            _ContainerSyntaxRule(path="request.container"),
        ),
        ContainerSurface.MCP: (
            _ContainerSyntaxRule(path="request.container"),
        ),
        ContainerSurface.A2A: (
            _ContainerSyntaxRule(path="request.container"),
        ),
        ContainerSurface.RUNTIME_ENVELOPE: (
            _ContainerSyntaxRule(path="runtime.container"),
        ),
        ContainerSurface.MODEL_BACKEND: (
            _ContainerSyntaxRule(path="model.container"),
        ),
    }
)


def container_syntax_diagnostics(
    surface: ContainerSurface | str,
    raw: Mapping[str, object],
) -> tuple[ContainerDiagnostic, ...]:
    assert isinstance(raw, Mapping), "raw must be a mapping"
    container_surface = _container_surface(surface)
    diagnostics: list[ContainerDiagnostic] = []
    for rule in _CONTAINER_SYNTAX_RULES.get(container_surface, ()):
        for path, value in _path_values(raw, rule.path):
            if rule.match_any_value or value == rule.expected_value:
                diagnostics.append(
                    _unsupported_syntax_diagnostic(
                        path,
                        surface=container_surface,
                    )
                )
    return tuple(diagnostics)


def assert_container_syntax_supported(
    surface: ContainerSurface | str,
    raw: Mapping[str, object],
) -> None:
    diagnostics = container_syntax_diagnostics(surface, raw)
    assert not diagnostics, "; ".join(
        f"{diagnostic.code.value} at {diagnostic.path}: {diagnostic.message}"
        for diagnostic in diagnostics
    )


def resolve_container_backend(
    settings: ContainerExecutionSettings,
    *,
    available_backends: Collection[ContainerBackend | str] = (),
) -> ContainerBackendResolution:
    assert isinstance(
        settings, ContainerExecutionSettings
    ), "settings must be container execution settings"
    selected_backend = _container_backend(settings.backend)
    available = frozenset(
        _container_backend(backend) for backend in available_backends
    )
    if selected_backend is ContainerBackend.NONE:
        if settings.required:
            return ContainerBackendResolution(
                backend=None,
                direct_execution_allowed=False,
                diagnostics=(
                    ContainerDiagnostic(
                        code=ContainerDiagnosticCode.BACKEND_REQUIRED,
                        path="container.backend",
                        message=(
                            "Container execution is required but no "
                            "container backend is selected."
                        ),
                        hint=(
                            "Select an authorized container backend or "
                            "disable the required container setting."
                        ),
                        category=ContainerDiagnosticCategory.RUNTIME,
                    ),
                ),
            )
        return ContainerBackendResolution(
            backend=None,
            direct_execution_allowed=True,
        )
    if selected_backend in available:
        return ContainerBackendResolution(
            backend=selected_backend,
            direct_execution_allowed=False,
        )
    return ContainerBackendResolution(
        backend=None,
        direct_execution_allowed=False,
        diagnostics=(
            ContainerDiagnostic(
                code=ContainerDiagnosticCode.BACKEND_UNAVAILABLE,
                path="container.backend",
                message="The selected container backend is unavailable.",
                hint=(
                    "Install or authorize the selected backend. Avalan must "
                    "not run this scope directly on the host as a fallback."
                ),
                category=ContainerDiagnosticCategory.RUNTIME,
            ),
        ),
    )


def _unsupported_syntax_diagnostic(
    path: str,
    *,
    surface: ContainerSurface,
) -> ContainerDiagnostic:
    return ContainerDiagnostic(
        code=ContainerDiagnosticCode.UNSUPPORTED_SYNTAX,
        path=path,
        message="Container syntax is not supported for this surface yet.",
        hint=(
            f"Remove {path} from {surface.value} until that container "
            "implementation phase lands."
        ),
        category=ContainerDiagnosticCategory.UNSUPPORTED,
    )


def _path_values(
    raw: Mapping[str, object],
    dotted_path: str,
) -> tuple[tuple[str, object], ...]:
    parts = tuple(dotted_path.split("."))
    matches: list[tuple[str, object]] = []
    _collect_path_values(raw, parts, "", matches)
    return tuple(matches)


def _collect_path_values(
    value: object,
    parts: tuple[str, ...],
    prefix: str,
    matches: list[tuple[str, object]],
) -> None:
    if not parts:
        matches.append((prefix, value))
        return
    if not isinstance(value, Mapping):
        return
    part = parts[0]
    remaining = parts[1:]
    if part == "*":
        for key, item in value.items():
            if isinstance(key, str):
                _collect_path_values(
                    item,
                    remaining,
                    _join_path(prefix, key),
                    matches,
                )
        return
    if part in value:
        _collect_path_values(
            value[part],
            remaining,
            _join_path(prefix, part),
            matches,
        )


def _join_path(prefix: str, part: str) -> str:
    if not prefix:
        return part
    return f"{prefix}.{part}"


def _container_backend(value: ContainerBackend | str) -> ContainerBackend:
    if isinstance(value, ContainerBackend):
        return value
    _assert_non_empty_string(value, "backend")
    assert isinstance(value, str)
    assert value in {
        backend.value for backend in ContainerBackend
    }, "backend contains unsupported value"
    return ContainerBackend(value)


def _container_scope(
    value: ContainerExecutionScope | str,
) -> ContainerExecutionScope:
    if isinstance(value, ContainerExecutionScope):
        return value
    _assert_non_empty_string(value, "scope")
    assert isinstance(value, str)
    assert value in {
        scope.value for scope in ContainerExecutionScope
    }, "scope contains unsupported value"
    return ContainerExecutionScope(value)


def _container_surface(value: ContainerSurface | str) -> ContainerSurface:
    if isinstance(value, ContainerSurface):
        return value
    _assert_non_empty_string(value, "surface")
    assert isinstance(value, str)
    assert value in {
        surface.value for surface in ContainerSurface
    }, "surface contains unsupported value"
    return ContainerSurface(value)
