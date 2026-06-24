from ..container import (
    ContainerExecutionScope,
    ContainerNormalizedRuntimeEnvelopePlan,
    ContainerPlanRequest,
    ContainerPlanRequestKind,
    ContainerRuntimeEnvelopeKind,
    ContainerSurface,
    ContainerToolRuntimeSettings,
    normalize_runtime_envelope_plan,
)
from .authority import remote_runtime_authority_key

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class RemoteContainerRequestError(ValueError):
    """Signal unsafe remote container runtime authority."""


@dataclass(frozen=True, kw_only=True, slots=True)
class RemoteContainerRequestPolicy:
    """Define operator-exposed remote container profile choices."""

    exposed_profiles: Sequence[str] = ()

    def __post_init__(self) -> None:
        profiles = tuple(self.exposed_profiles)
        assert len(set(profiles)) == len(
            profiles
        ), "exposed profiles must be unique"
        for profile in profiles:
            assert isinstance(profile, str), "exposed profile must be a string"
            assert profile.strip(), "exposed profile cannot be empty"
            assert (
                profile == profile.strip()
            ), "exposed profile cannot include surrounding whitespace"
        object.__setattr__(self, "exposed_profiles", profiles)

    def exposes(self, profile: str) -> bool:
        """Return whether the remote profile is exposed."""
        return profile in self.exposed_profiles


@dataclass(frozen=True, kw_only=True, slots=True)
class RemoteContainerRequest:
    """Represent sanitized remote request container input."""

    arguments: dict[str, object]
    profile: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class ServerRuntimeEnvelopeStatus:
    """Represent trusted server runtime envelope planning diagnostics."""

    plan: ContainerNormalizedRuntimeEnvelopePlan | None = None
    diagnostics: tuple[dict[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.plan is not None:
            assert isinstance(
                self.plan,
                ContainerNormalizedRuntimeEnvelopePlan,
            )
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, dict)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def ok(self) -> bool:
        """Return whether the trusted server envelope is available."""
        return not self.diagnostics


_SAFE_CONTAINER_PROFILE_KEYS = ("container_profile", "containerProfile")
_EXTRA_RUNTIME_AUTHORITY_KEYS = frozenset(
    {
        "capabilities",
        "capability",
        "commandmode",
        "containerflags",
        "env",
        "environment",
        "environmentvariables",
        "envvars",
        "gid",
        "platform",
        "privileged",
        "pullpolicy",
        "readonlyrootfs",
        "uid",
        "user",
        "workdir",
        "workspace",
        "workspaceroot",
        "workingdirectory",
    }
)


def remote_container_policy_from_state(
    state: object,
) -> RemoteContainerRequestPolicy | None:
    """Return the trusted remote container policy from app state."""
    policy = getattr(state, "remote_container_policy", None)
    if policy is None:
        return None
    assert isinstance(policy, RemoteContainerRequestPolicy)
    return policy


def remote_container_policy_from_runtime_settings(
    runtime_settings: ContainerToolRuntimeSettings | None,
) -> RemoteContainerRequestPolicy | None:
    """Return the remote profile policy exposed by trusted settings."""
    if runtime_settings is None:
        return None
    assert isinstance(runtime_settings, ContainerToolRuntimeSettings)
    effective_settings = runtime_settings.effective_settings
    if effective_settings is None:
        return None
    if not effective_settings.enabled:
        return None
    if not effective_settings.source.can_define_runtime_authority:
        return None
    if effective_settings.profile_name is None:
        return None
    return RemoteContainerRequestPolicy(
        exposed_profiles=(effective_settings.profile_name,)
    )


def server_runtime_envelope_status_from_runtime_settings(
    runtime_settings: ContainerToolRuntimeSettings | None,
    *,
    server_name: str = "server",
    request_id: str = "server",
    envelope_runtime_available: bool = False,
    readiness_timeout_seconds: int = 30,
) -> ServerRuntimeEnvelopeStatus:
    """Return trusted server runtime envelope plan and diagnostics."""
    assert isinstance(server_name, str) and server_name.strip()
    assert isinstance(request_id, str) and request_id.strip()
    assert isinstance(envelope_runtime_available, bool)
    effective_settings = (
        None
        if runtime_settings is None
        else runtime_settings.effective_settings
    )
    if effective_settings is None or not effective_settings.enabled:
        return ServerRuntimeEnvelopeStatus()
    if (
        effective_settings.scope
        is not ContainerExecutionScope.RUNTIME_ENVELOPE
        or effective_settings.source.surface is not ContainerSurface.SERVER
    ):
        return ServerRuntimeEnvelopeStatus()
    plan = normalize_runtime_envelope_plan(
        effective_settings,
        ContainerPlanRequest(
            request_kind=ContainerPlanRequestKind.SERVER,
            logical_name=server_name,
            command="avalan-server",
            argv=("avalan", "server"),
            cwd="/workspace",
            scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
            request_id=request_id,
        ),
        envelope_kind=ContainerRuntimeEnvelopeKind.SERVER,
        readiness_timeout_seconds=readiness_timeout_seconds,
    )
    if envelope_runtime_available:
        return ServerRuntimeEnvelopeStatus(plan=plan)
    return ServerRuntimeEnvelopeStatus(
        plan=plan,
        diagnostics=(
            {
                "code": "server.runtime_envelope_unavailable",
                "path": "runtime.container.envelope",
                "message": (
                    "Server runtime envelope execution is not available "
                    "inside this server process."
                ),
                "hint": (
                    "Start the server through a trusted envelope-aware "
                    "operator runtime."
                ),
            },
        ),
    )


def validate_remote_container_arguments(
    arguments: Mapping[str, object],
    *,
    policy: RemoteContainerRequestPolicy | None = None,
) -> RemoteContainerRequest:
    """Reject remote container authority and return sanitized arguments."""
    assert isinstance(arguments, Mapping)
    sanitized = dict(arguments)
    profile = _pop_profile_selection(sanitized, policy)
    _reject_container_authority(sanitized, path="arguments")
    return RemoteContainerRequest(arguments=sanitized, profile=profile)


def _pop_profile_selection(
    arguments: dict[str, object],
    policy: RemoteContainerRequestPolicy | None,
) -> str | None:
    profiles: list[str] = []
    if "container" in arguments:
        profiles.append(
            _container_profile_from_envelope(
                arguments.pop("container"),
                policy,
            )
        )

    for key in _SAFE_CONTAINER_PROFILE_KEYS:
        if key in arguments:
            profiles.append(
                _validated_remote_profile(arguments.pop(key), policy)
            )

    if len(profiles) > 1:
        raise RemoteContainerRequestError(
            "Remote request must use one container profile selector"
        )
    return profiles[0] if profiles else None


def _container_profile_from_envelope(
    value: object,
    policy: RemoteContainerRequestPolicy | None,
) -> str:
    if not isinstance(value, Mapping):
        raise RemoteContainerRequestError(
            "Remote request container selector must be an object"
        )
    if set(value) != {"profile"}:
        raise RemoteContainerRequestError(
            "Remote request container envelope can only select a profile"
        )
    return _validated_remote_profile(value["profile"], policy)


def _validated_remote_profile(
    value: object,
    policy: RemoteContainerRequestPolicy | None,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RemoteContainerRequestError(
            "Remote container profile selector must be a non-empty string"
        )
    profile = value.strip()
    if policy is None or not policy.exposes(profile):
        raise RemoteContainerRequestError(
            f'Remote container profile "{profile}" is not exposed'
        )
    return profile


def _reject_container_authority(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_path = f"{path}.{key}"
            if _is_container_authority_key(key):
                raise RemoteContainerRequestError(
                    "Remote request cannot supply container runtime "
                    f"authority at {key_path}"
                )
            _reject_container_authority(item, path=key_path)
        return

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _reject_container_authority(item, path=f"{path}[{index}]")


def _is_container_authority_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = _normalized_key(key)
    return (
        remote_runtime_authority_key(key)
        or normalized in _EXTRA_RUNTIME_AUTHORITY_KEYS
    )


def _normalized_key(key: str) -> str:
    return "".join(
        character.lower() for character in key if character not in {"-", "_"}
    )
