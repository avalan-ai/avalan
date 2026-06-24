from collections.abc import Mapping, Sequence
from re import sub
from typing import cast
from urllib.parse import parse_qsl, urlsplit

REMOTE_CONTAINER_PROFILE_SELECTOR_KEYS = frozenset(
    {
        "container",
        "containerProfile",
        "container_profile",
    }
)

_REMOTE_RUNTIME_AUTHORITY_KEYS = frozenset(
    {
        "backend",
        "backendflag",
        "backendflags",
        "backendoption",
        "backendoptions",
        "container",
        "containerpolicy",
        "containerprofile",
        "containerprofiles",
        "containersettings",
        "containerruntime",
        "containerruntimepolicy",
        "buildpolicy",
        "capabilities",
        "capability",
        "commandmode",
        "containerflags",
        "device",
        "devices",
        "devicerequest",
        "devicerequests",
        "egress",
        "egressallowlist",
        "env",
        "environment",
        "environmentvariable",
        "environmentvariables",
        "envvar",
        "envvars",
        "gid",
        "image",
        "imageref",
        "imagereference",
        "images",
        "memorybytes",
        "mount",
        "mountpath",
        "mountpaths",
        "mounts",
        "network",
        "networkmode",
        "networkpolicy",
        "networks",
        "pids",
        "platform",
        "policyversion",
        "privileged",
        "pullpolicy",
        "readonlyrootfs",
        "resource",
        "resourcelimit",
        "resourcelimits",
        "resources",
        "runtime",
        "runtimecontainer",
        "runtimeenvelope",
        "runtimelimits",
        "runtimepolicy",
        "runtimeprofile",
        "runtimeprofiles",
        "secret",
        "secretdelivery",
        "secretdeliveries",
        "secrets",
        "timeoutseconds",
        "uid",
        "user",
        "workdir",
        "workingdirectory",
        "workspace",
        "workspaceroot",
    }
)
_REMOTE_RUNTIME_AUTHORITY_PREFIXES = ("container", "runtime")


def remote_runtime_authority_key(key: object) -> bool:
    normalized = _normalize_authority_key(key)
    if not normalized:
        return False
    if normalized in _REMOTE_RUNTIME_AUTHORITY_KEYS:
        return True
    if normalized.startswith(_REMOTE_RUNTIME_AUTHORITY_PREFIXES):
        return True
    if normalized.endswith("backend"):
        return True
    return "secret" in normalized


def reject_remote_runtime_authority_fields(
    value: object,
    *,
    path: str = "request",
    skip_keys: set[str] | frozenset[str] = frozenset(),
) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            item_path = f"{path}.{key}"
            if key in skip_keys:
                continue
            if remote_runtime_authority_key(key):
                raise ValueError(_authority_error(item_path, key))
            reject_remote_runtime_authority_fields(
                item,
                path=item_path,
                skip_keys=skip_keys,
            )
        return
    if _is_sequence(value):
        for index, item in enumerate(cast(Sequence[object], value)):
            reject_remote_runtime_authority_fields(
                item,
                path=f"{path}[{index}]",
                skip_keys=skip_keys,
            )


def reject_remote_runtime_authority_extra_fields(
    value: object,
    *,
    allowed_fields: set[str] | frozenset[str],
    allow_container_profile_selector: bool = False,
    path: str = "request",
) -> None:
    if not isinstance(value, Mapping):
        return
    for raw_key, item in value.items():
        key = str(raw_key)
        if key in allowed_fields:
            continue
        item_path = f"{path}.{key}"
        if allow_container_profile_selector and _is_profile_selector(
            key,
            item,
        ):
            continue
        if remote_runtime_authority_key(key):
            raise ValueError(_authority_error(item_path, key))
        reject_remote_runtime_authority_fields(item, path=item_path)


def reject_remote_runtime_authority_model_identifier(
    value: object,
    *,
    path: str = "request.model",
) -> None:
    if not isinstance(value, str) or "?" not in value:
        return
    for key, _item in parse_qsl(
        urlsplit(value).query,
        keep_blank_values=True,
    ):
        if remote_runtime_authority_key(key):
            raise ValueError(_authority_error(f"{path}?{key}", key))


def _authority_error(path: str, key: str) -> str:
    return (
        "Remote requests cannot provide runtime authority field "
        f"'{key}' at {path}"
    )


def _normalize_authority_key(key: object) -> str:
    return sub(r"[^a-z0-9]", "", str(key).lower())


def _is_profile_selector(key: str, value: object) -> bool:
    if key == "container":
        return (
            isinstance(value, Mapping)
            and set(value) == {"profile"}
            and _is_non_empty_string(value["profile"])
        )
    if key in {"containerProfile", "container_profile"}:
        return _is_non_empty_string(value)
    return False


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (bytes, bytearray, memoryview, str),
    )
