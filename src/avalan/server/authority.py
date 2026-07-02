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
        "environmentpolicy",
        "environmentvariable",
        "environmentvariables",
        "envvar",
        "envvars",
        "gid",
        "approval",
        "approvalpolicy",
        "approvalrecord",
        "approvalrecords",
        "approvals",
        "approvedby",
        "allowedroot",
        "allowedroots",
        "deniedroot",
        "deniedroots",
        "denyroot",
        "denyroots",
        "executionmode",
        "hostroot",
        "hostroots",
        "image",
        "imageref",
        "imagereference",
        "images",
        "inputroot",
        "inputroots",
        "isolation",
        "isolationbackend",
        "isolationconfig",
        "isolationmode",
        "isolationplan",
        "isolationpolicy",
        "isolationprofile",
        "isolationprofiles",
        "isolationruntime",
        "isolationsettings",
        "memorybytes",
        "mode",
        "mount",
        "mountpath",
        "mountpaths",
        "mounts",
        "mountroot",
        "mountroots",
        "network",
        "networkmode",
        "networkpolicy",
        "networks",
        "outputroot",
        "outputroots",
        "pids",
        "platform",
        "policyversion",
        "privileged",
        "pullpolicy",
        "readroot",
        "readroots",
        "readonlyrootfs",
        "resource",
        "resourcelimit",
        "resourcelimits",
        "resources",
        "reviewmode",
        "rootpath",
        "rootpaths",
        "roots",
        "runtime",
        "runtimecontainer",
        "runtimeenvelope",
        "runtimelimits",
        "runtimepolicy",
        "runtimeprofile",
        "runtimeprofiles",
        "sandbox",
        "sandboxbackend",
        "sandboxconfig",
        "sandboxmode",
        "sandboxpolicy",
        "sandboxprofile",
        "sandboxprofileid",
        "sandboxprofiles",
        "sandboxroot",
        "sandboxroots",
        "sandboxruntime",
        "sandboxsettings",
        "secret",
        "secretdelivery",
        "secretdeliveries",
        "secrets",
        "scratchroot",
        "scratchroots",
        "temproot",
        "temproots",
        "timeoutseconds",
        "uid",
        "user",
        "workdir",
        "workingdirectory",
        "workspace",
        "workspaceroot",
        "writeroot",
        "writeroots",
    }
)
_REMOTE_RUNTIME_AUTHORITY_PREFIXES = (
    "container",
    "isolation",
    "runtime",
    "sandbox",
)
_REMOTE_SHELL_AUTHORITY_KEYS = frozenset({"shell"})
_REMOTE_SHELL_AUTHORITY_PREFIXES = ("shellruntime",)
_REMOTE_SHELL_AUTHORITY_MARKERS = (
    "allowpipeline",
    "allowpipelines",
    "allowshell",
)
_REMOTE_SKILL_AUTHORITY_KEYS = frozenset(
    {
        "allowhiddenpaths",
        "authoritykind",
        "authoritykinds",
        "bootstrapenabled",
        "cursorlimits",
        "indexlimits",
        "loadskills",
        "maxactivecursors",
        "maxbytesperread",
        "maxcursorageseconds",
        "maxdirectoryentriespersource",
        "maxfilespersource",
        "maxindexedbytes",
        "maxlinesperread",
        "maxresourcesperskill",
        "maxresourcespersource",
        "maxskills",
        "maxsourcedepth",
        "maxsources",
        "modelfacingload",
        "modelfacingloadbehavior",
        "packagepath",
        "packagepaths",
        "readlimits",
        "registry",
        "registries",
        "registrymutation",
        "registrymutations",
        "registryversion",
        "rootpath",
        "rootpaths",
        "skillconfig",
        "skillconfiguration",
        "skillload",
        "skillloadbehavior",
        "skillregistry",
        "skillregistries",
        "skillsettings",
        "skills",
        "skillsconfig",
        "skillsconfiguration",
        "skillsload",
        "skillsloadbehavior",
        "skillsregistry",
        "skillsregistries",
        "skillssettings",
        "sourceauthorities",
        "sourceauthority",
        "sourcelabels",
        "sourcelimits",
        "sourceroot",
        "sourceroots",
        "sources",
    }
)
_REMOTE_SKILL_AUTHORITY_PREFIXES = (
    "skillregistry",
    "skillsregistry",
    "skillsettings",
    "skillssettings",
    "skillsource",
    "skillssource",
)
_JSON_SCHEMA_DECLARATION_MAP_KEYS = frozenset(
    {
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
        "properties",
    }
)
_JSON_SCHEMA_VALUE_KEYS = frozenset(
    {
        "additionalProperties",
        "contains",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_JSON_SCHEMA_SEQUENCE_VALUE_KEYS = frozenset(
    {
        "allOf",
        "anyOf",
        "oneOf",
        "prefixItems",
    }
)


def remote_runtime_authority_key(key: object) -> bool:
    normalized = _normalize_authority_key(key)
    if not normalized:
        return False
    if normalized in _REMOTE_RUNTIME_AUTHORITY_KEYS:
        return True
    if normalized in _REMOTE_SHELL_AUTHORITY_KEYS:
        return True
    if normalized in _REMOTE_SKILL_AUTHORITY_KEYS:
        return True
    if normalized.startswith(_REMOTE_SHELL_AUTHORITY_PREFIXES):
        return True
    if normalized.startswith(_REMOTE_SKILL_AUTHORITY_PREFIXES):
        return True
    if any(marker in normalized for marker in _REMOTE_SHELL_AUTHORITY_MARKERS):
        return True
    if normalized.startswith("toolshell"):
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
    _reject_remote_runtime_authority_fields(
        value,
        path=path,
        skip_keys=skip_keys,
        schema_context=False,
    )


def _reject_remote_runtime_authority_fields(
    value: object,
    *,
    path: str,
    skip_keys: set[str] | frozenset[str],
    schema_context: bool,
) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            item_path = f"{path}.{key}"
            if key in skip_keys:
                continue
            enters_schema_context = _enters_json_schema_context(
                key,
                path,
                item,
            )
            item_schema_context = schema_context or enters_schema_context
            if _is_json_schema_declaration_map(
                key,
                item,
                item_schema_context,
            ):
                _reject_json_schema_property_definitions(
                    item,
                    path=item_path,
                    skip_keys=skip_keys,
                )
                continue
            if remote_runtime_authority_key(key):
                raise ValueError(_authority_error(item_path, key))
            _reject_remote_runtime_authority_fields(
                item,
                path=item_path,
                skip_keys=skip_keys,
                schema_context=enters_schema_context
                or _child_schema_context(key, item_schema_context),
            )
        return
    if _is_sequence(value):
        for index, item in enumerate(cast(Sequence[object], value)):
            _reject_remote_runtime_authority_fields(
                item,
                path=f"{path}[{index}]",
                skip_keys=skip_keys,
                schema_context=schema_context,
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
        item_path = f"{path}.{key}"
        if key in allowed_fields:
            reject_remote_runtime_authority_fields(item, path=item_path)
            continue
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


def _is_json_schema_declaration_map(
    key: str,
    value: object,
    schema_context: bool,
) -> bool:
    return (
        schema_context
        and key in _JSON_SCHEMA_DECLARATION_MAP_KEYS
        and isinstance(value, Mapping)
    )


def _enters_json_schema_context(
    key: str,
    path: str,
    value: object,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    if key == "parameters":
        return path.startswith(("chat.tools[", "request.tools[")) and (
            path.endswith(".function")
        )
    if key != "schema":
        return False
    return path in {
        "chat.response_format",
        "chat.response_format.json_schema",
        "request.response_format",
        "request.response_format.json_schema",
        "request.text.format",
        "responses.response_format",
        "responses.response_format.json_schema",
        "responses.text.format",
    }


def _child_schema_context(key: str, schema_context: bool) -> bool:
    return schema_context and (
        key in _JSON_SCHEMA_VALUE_KEYS
        or key in _JSON_SCHEMA_SEQUENCE_VALUE_KEYS
    )


def _reject_json_schema_property_definitions(
    value: Mapping[object, object],
    *,
    path: str,
    skip_keys: set[str] | frozenset[str],
) -> None:
    for raw_key, item in value.items():
        _reject_remote_runtime_authority_fields(
            item,
            path=f"{path}.{raw_key}",
            skip_keys=skip_keys,
            schema_context=True,
        )


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
